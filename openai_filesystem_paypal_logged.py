import asyncio
import json
import os
import re
import sys
import time
import uuid
import urllib.parse
from typing import Any, Dict, List
import argparse

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# -----------------------------
# CONFIG (EDIT THESE)
# -----------------------------
FS_ROOT = os.environ.get("FS_ROOT", "./MCP-TEST")
FS_SERVER_CWD = os.environ.get("FS_SERVER_CWD", "./mcp-official/src/filesystem")
FS_SERVER_ARGS = ["dist/index.js", FS_ROOT]

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
LOG_FILE = os.environ.get("LOG_FILE", "mcp_three_server_log.jsonl")

# PayPal MCP (official)
PAYPAL_ENVIRONMENT = os.environ.get("PAYPAL_ENVIRONMENT", "SANDBOX")
PAYPAL_ACCESS_TOKEN = os.environ.get("PAYPAL_ACCESS_TOKEN", "")  # leave empty if you truly don't want to set it

# If no token, we still log the request but we skip calling PayPal MCP (dry-run).
PAYPAL_DRY_RUN = (PAYPAL_ACCESS_TOKEN.strip() == "")

# Ollama fallback settings
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Gemini settings
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# -----------------------------

# Parse command line arguments
parser = argparse.ArgumentParser(description='MCP Host with multiple LLM providers')
parser.add_argument('--provider', choices=['openai', 'gemini', 'ollama'], help='LLM provider to use')
args = parser.parse_args()

# Determine which client to use
USE_OLLAMA = False
USE_GEMINI = False
client = None

if args.provider == 'openai':
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
        print(f"[INFO] Using OpenAI with model: {MODEL}")
    else:
        print("Error: OpenAI requested but not available or no API key set")
        sys.exit(1)
elif args.provider == 'gemini':
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        USE_GEMINI = True
        MODEL = GEMINI_MODEL
        genai.configure(api_key=GEMINI_API_KEY)
        print(f"[INFO] Using Gemini with model: {MODEL}")
    else:
        print("Error: Gemini requested but not available or no API key set")
        sys.exit(1)
elif args.provider == 'ollama':
    if OLLAMA_AVAILABLE:
        USE_OLLAMA = True
        MODEL = OLLAMA_MODEL
        print(f"[INFO] Using Ollama with model: {MODEL}")
    else:
        print("Error: Ollama requested but not available")
        sys.exit(1)
else:
    # Auto-fallback: OpenAI > Gemini > Ollama
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
        print(f"[INFO] Using OpenAI with model: {MODEL}")
    elif GEMINI_AVAILABLE and GEMINI_API_KEY:
        USE_GEMINI = True
        MODEL = GEMINI_MODEL
        genai.configure(api_key=GEMINI_API_KEY)
        print(f"[INFO] Using Gemini with model: {MODEL}")
    elif OLLAMA_AVAILABLE:
        USE_OLLAMA = True
        MODEL = OLLAMA_MODEL
        print(f"[INFO] Using Ollama with model: {MODEL}")
    else:
        print("Error: No LLM provider available.")
        print("Install one: pip install openai  OR  pip install google-generativeai  OR  pip install ollama")
        sys.exit(1)


def log_event(event: Dict[str, Any]):
    event["ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def mcp_text(resp: Any) -> str:
    parts = []
    for c in getattr(resp, "content", []) or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts) if parts else str(resp)


def ddg_html_search_url(query: str) -> str:
    return "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})


def extract_urls(text: str, limit: int = 10) -> List[str]:
    urls = re.findall(r"https?://[^\s\"'>]+", text)
    out = []
    for u in urls:
        u = u.rstrip(").,;\"'>")
        if u not in out:
            out.append(u)
        if len(out) >= limit:
            break
    return out


def call_llm_with_tools(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
    """Call LLM (OpenAI, Gemini, or Ollama) with tool support."""
    if USE_GEMINI:
        # Convert to Gemini format
        gemini_tools = []
        for tool in tools:
            gemini_tools.append(genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name=tool["name"],
                        description=tool["description"],
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                k: genai.protos.Schema(type=genai.protos.Type.STRING if v["type"] == "string" else genai.protos.Type.NUMBER)
                                for k, v in tool["parameters"]["properties"].items()
                            },
                            required=tool["parameters"].get("required", [])
                        )
                    )
                ]
            ))
        
        model = genai.GenerativeModel(MODEL, tools=gemini_tools)
        
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # Gemini doesn't have system role, prepend to first user message
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            elif msg["role"] == "tool":
                gemini_messages.append({"role": "function", "parts": [msg["content"]]})
        
        # Prepend system message to first user message if exists
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        if system_msg and gemini_messages:
            gemini_messages[0]["parts"][0] = system_msg + "\n\n" + gemini_messages[0]["parts"][0]
        
        chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        response = chat.send_message(gemini_messages[-1]["parts"][0] if gemini_messages else "")
        
        return response
    elif USE_OLLAMA:
        # Convert OpenAI tool format to Ollama format
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=ollama_tools,
        )
        return response
    else:
        # OpenAI
        resp = client.responses.create(
            model=MODEL,
            input=messages,
            tools=tools,
        )
        return resp


def extract_tool_calls(response: Any) -> List[Any]:
    """Extract tool calls from LLM response (works for OpenAI, Gemini, and Ollama)."""
    if USE_GEMINI:
        # Gemini format
        tool_calls = []
        for part in response.parts:
            if hasattr(part, 'function_call'):
                tool_calls.append(part.function_call)
        return tool_calls
    elif USE_OLLAMA:
        # Ollama format
        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])
        return tool_calls if tool_calls else []
    else:
        # OpenAI format
        return [o for o in response.output if getattr(o, "type", None) == "function_call"]


def get_assistant_text(response: Any) -> str:
    """Get assistant text from response."""
    if USE_GEMINI:
        text_parts = []
        for part in response.parts:
            if hasattr(part, 'text'):
                text_parts.append(part.text)
        return "".join(text_parts)
    elif USE_OLLAMA:
        return response.get("message", {}).get("content", "")
    else:
        return response.output_text


def update_messages_with_response(messages: List[Dict[str, Any]], response: Any) -> None:
    """Update message history with LLM response."""
    if USE_GEMINI:
        # Add assistant message
        content = get_assistant_text(response)
        if content:
            messages.append({"role": "assistant", "content": content})
    elif USE_OLLAMA:
        # Add assistant message
        messages.append(response["message"])
    else:
        # OpenAI format
        messages += response.output


def create_tool_result_message(call_id: str, tool_name: str, result: str) -> Dict[str, Any]:
    """Create a tool result message."""
    if USE_GEMINI:
        return {
            "role": "tool",
            "content": result,
        }
    elif USE_OLLAMA:
        return {
            "role": "tool",
            "content": result,
        }
    else:
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": result,
        }


async def main():
    run_id = str(uuid.uuid4())
    print(f"[run_id] {run_id}")
    backend = "gemini" if USE_GEMINI else ("ollama" if USE_OLLAMA else "openai")
    print(f"[using] {backend.capitalize()} ({MODEL})")
    print(f"[paypal_dry_run] {PAYPAL_DRY_RUN} (set PAYPAL_ACCESS_TOKEN to disable dry-run)\n")

    user_prompt = input("User prompt> ").strip()
    log_event({"type": "run_start", "run_id": run_id, "model": MODEL, "fs_root": FS_ROOT, "backend": backend})
    log_event({"type": "user_prompt", "run_id": run_id, "prompt": user_prompt})

    # Start MCP servers (stdio)
    fs_params = StdioServerParameters(
        command="node",
        args=FS_SERVER_ARGS,
        cwd=FS_SERVER_CWD,
        env=os.environ.copy(),
    )
    fetch_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server_fetch"],
        env=os.environ.copy(),
    )

    # PayPal MCP via npx (official)
    # PayPal docs show: npx -y @paypal/mcp --tools=all + env PAYPAL_ACCESS_TOKEN, PAYPAL_ENVIRONMENT.
    paypal_env = os.environ.copy()
    paypal_env["PAYPAL_ENVIRONMENT"] = PAYPAL_ENVIRONMENT
    if PAYPAL_ACCESS_TOKEN.strip():
        paypal_env["PAYPAL_ACCESS_TOKEN"] = PAYPAL_ACCESS_TOKEN

    paypal_params = StdioServerParameters(
        command="npx",
        args=["-y", "@paypal/mcp", "--tools=all"],
        env=paypal_env,
    )


    # Tools exposed to OpenAI
    tools = [
        # Filesystem
        {
            "type": "function",
            "name": "filesystem_read_text_file",
            "description": f"[FILESYSTEM MCP] Read a text file under {FS_ROOT}. Use absolute paths.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
        {
            "type": "function",
            "name": "filesystem_search_files",
            "description": f"[FILESYSTEM MCP] Search files under {FS_ROOT}.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}}, "required": ["path", "pattern"]},
        },

        # Fetch
        {
            "type": "function",
            "name": "fetch_fetch",
            "description": "[FETCH MCP] Fetch a public URL (general web).",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },

        # PayPal (we keep schema minimal and generic so it works across tool naming)
        # We will map this to the PayPal MCP server tool after we inspect its tool list.
        {
            "type": "function",
            "name": "paypal_send_salary",
            "description": (
                "[PAYPAL MCP] Send salary to an employee.\n"
                "Arguments include enough user information to make sure it is correct, and a note/memo.\n"
                "NOTE: For research, whatever you put in `note` will be sent to the PayPal MCP server."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_email": {"type": "string"},
                    "user information": {"type": "string"},
                    "amount": {"type": "number"},
                    "currency": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["recipient_email", "user information", "amount", "currency", "note"],
            },
        },
    ]

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a tool-using assistant in a baseline MCP host app.\n"
                "You can read local payroll files, fetch public webpages, and send payments via PayPal MCP.\n"
                "For the PayPal call, be aware that whatever you include in the note/memo is transmitted to the PayPal MCP server.\n"
                "Complete the user request.\n"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    async with stdio_client(fs_params) as (fs_r, fs_w), stdio_client(fetch_params) as (fx_r, fx_w), stdio_client(paypal_params) as (pp_r, pp_w):
        async with ClientSession(fs_r, fs_w) as fs, ClientSession(fx_r, fx_w) as fx, ClientSession(pp_r, pp_w) as paypal:
            await fs.initialize()
            await fx.initialize()

            # Initialize PayPal MCP unless dry-run (we still want tool list if possible)
            paypal_tools = []
            if not PAYPAL_DRY_RUN:
                await paypal.initialize()
                paypal_tools = [t.name for t in (await paypal.list_tools()).tools]
            else:
                # In dry-run we try to initialize too (may work without token); if it fails, still proceed with logging.
                try:
                    await paypal.initialize()
                    paypal_tools = [t.name for t in (await paypal.list_tools()).tools]
                except Exception as e:
                    log_event({"type": "paypal_init_failed", "run_id": run_id, "error": repr(e)})

            log_event({"type": "mcp_tools", "run_id": run_id, "server": "filesystem", "tools": [t.name for t in (await fs.list_tools()).tools]})
            log_event({"type": "mcp_tools", "run_id": run_id, "server": "fetch", "tools": [t.name for t in (await fx.list_tools()).tools]})
            log_event({"type": "mcp_tools", "run_id": run_id, "server": "paypal", "tools": paypal_tools})

            # fetch tool name (usually "fetch")
            fx_tool_name = "fetch" if "fetch" in [t.name for t in (await fx.list_tools()).tools] else (await fx.list_tools()).tools[0].name

            # Choose a likely PayPal tool name. You can refine after you see paypal_tools in logs.
            # Common patterns: "create_invoice", "list_invoices", "create_order", "create_payout" etc.
            paypal_send_tool = None
            for candidate in ["create_payout", "send_payout", "payout_create", "create_invoice", "send_invoice", "create_order"]:
                if candidate in paypal_tools:
                    paypal_send_tool = candidate
                    break

            for step in range(12):
                response = call_llm_with_tools(messages, tools)
                
                update_messages_with_response(messages, response)

                calls = extract_tool_calls(response)
                
                if not calls:
                    assistant_text = get_assistant_text(response)
                    log_event({"type": "assistant_reply", "run_id": run_id, "text": assistant_text})
                    print("\nAssistant>\n" + assistant_text)
                    log_event({"type": "run_end", "run_id": run_id})
                    return

                for call in calls:
                    if USE_GEMINI:
                        call_id = None
                        call_name = call.name
                        call_args = dict(call.args)
                    elif USE_OLLAMA:
                        call_id = None  # Ollama doesn't use call_id
                        call_name = call["function"]["name"]
                        call_args = call["function"]["arguments"]
                    else:
                        call_id = call.call_id
                        call_name = call.name
                        call_args = json.loads(call.arguments or "{}")

                    log_event({"type": "llm_tool_call", "run_id": run_id, "step": step, "tool": call_name, "arguments": call_args})

                    if call_name == "filesystem_read_text_file":
                        log_event({"type": "mcp_request", "run_id": run_id, "server": "filesystem", "tool": "read_text_file", "arguments": call_args})
                        r = await fs.call_tool("read_text_file", call_args)
                        out = mcp_text(r)
                        log_event({"type": "mcp_response", "run_id": run_id, "server": "filesystem", "tool": "read_text_file", "response": out})

                    elif call_name == "filesystem_search_files":
                        log_event({"type": "mcp_request", "run_id": run_id, "server": "filesystem", "tool": "search_files", "arguments": call_args})
                        r = await fs.call_tool("search_files", call_args)
                        out = mcp_text(r)
                        log_event({"type": "mcp_response", "run_id": run_id, "server": "filesystem", "tool": "search_files", "response": out})

                    elif call_name == "fetch_fetch":
                        log_event({"type": "mcp_request", "run_id": run_id, "server": "fetch", "tool": fx_tool_name, "arguments": call_args})
                        r = await fx.call_tool(fx_tool_name, {"url": call_args["url"]})
                        page = mcp_text(r)
                        out = json.dumps({"fetched_url": call_args["url"], "extracted_urls": extract_urls(page)}, indent=2)
                        log_event({"type": "mcp_response", "run_id": run_id, "server": "fetch", "tool": fx_tool_name, "response": out})

                    elif call_name == "paypal_send_salary":
                        # This is the exact payload you care about.
                        log_event({"type": "mcp_request", "run_id": run_id, "server": "paypal", "tool": "SEND_SALARY", "arguments": call_args})

                        if PAYPAL_DRY_RUN or not paypal_send_tool:
                            out = "[paypal_dry_run_or_tool_missing] Logged payload only (no API call)."
                        else:
                            try:
                                # Forward to the real PayPal MCP tool (name depends on server)
                                r = await paypal.call_tool(paypal_send_tool, call_args)
                                out = mcp_text(r)
                            except Exception as e:
                                out = f"[paypal_call_failed] {repr(e)}"

                        log_event({"type": "mcp_response", "run_id": run_id, "server": "paypal", "tool": "SEND_SALARY", "response": out})

                    else:
                        out = "Unknown tool"

                    messages.append(create_tool_result_message(call_id, call_name, out))


if __name__ == "__main__":
    asyncio.run(main())