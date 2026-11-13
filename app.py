"""ServiceMate: AI service agent (FastAPI + Gemini + MCP tools + RAG)."""
import asyncio
import json
import os
import sys
from collections import OrderedDict
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

BASE = Path(__file__).parent
IST = timezone(timedelta(hours=5, minutes=30))
MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]  # second one is the fallback
MAX_TOOL_ROUNDS = 6
MAX_SESSIONS = 200

SYSTEM_PROMPT = """You are ServiceMate, the AI assistant of ServiceMate Home Services (AC repair,
plumbing, electrician and deep cleaning in Jaipur).

Today's date is {today} ({weekday}), India time.

How you work:
- Reply in the language the customer uses (Hindi, English or Hinglish). Keep replies short and friendly.
- For ANY question about prices, policies, warranty, coverage, timings or FAQs, first call
  search_knowledge_base (write the query in English) and answer only from what it returns. Never invent prices or policies.
  If the knowledge base has no answer, say so and suggest contacting the team.
- To book: find out the service and preferred date, call check_availability, offer the free slots,
  then collect the customer's name and 10-digit phone number. Read all details back and ask for a
  clear yes. Only after the yes, call create_booking. Share the booking ID from the tool result.
- Convert words like "kal" or "next Monday" into a YYYY-MM-DD date using today's date.
- Use get_booking when the customer asks about an existing booking ID.
- Do not give medical, legal or financial advice. Stay on the topic of our services.
"""

state = {"mcp": None, "declarations": [], "client": None}
sessions: "OrderedDict[str, list]" = OrderedDict()


def _schema(prop: dict) -> types.Schema:
    kind = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}.get(
        prop.get("type"), "STRING"
    )
    return types.Schema(type=kind, description=prop.get("description", ""))


def _declaration(tool) -> types.FunctionDeclaration:
    schema = tool.inputSchema or {}
    props = schema.get("properties", {})
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or "",
        parameters=types.Schema(
            type="OBJECT",
            properties={k: _schema(v) for k, v in props.items()},
            required=schema.get("required", []),
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["client"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    async with AsyncExitStack() as stack:
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(BASE / "mcp_server.py")],
            env={**os.environ},  # the MCP subprocess needs GEMINI_API_KEY for embeddings
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = (await session.list_tools()).tools
        state["mcp"] = session
        state["declarations"] = [_declaration(t) for t in tools]
        print("MCP tools loaded:", [t.name for t in tools])
        yield


app = FastAPI(title="ServiceMate Agent", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str
    message: str


async def _generate(contents, system_text):
    """Call Gemini; retry on busy errors and fall back to a lighter model."""
    config = types.GenerateContentConfig(
        system_instruction=system_text,
        tools=[types.Tool(function_declarations=state["declarations"])],
    )
    last = None
    for model in MODELS:
        for _ in range(2):
            try:
                return await state["client"].aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except genai_errors.APIError as e:
                last = e
                if getattr(e, "code", None) in (429, 500, 503, 504):
                    await asyncio.sleep(1.5)
                    continue
                raise
    raise last


async def _call_tool(name: str, args: dict) -> str:
    result = await state["mcp"].call_tool(name, args)
    text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
    return text or "(empty result)"


@app.post("/api/chat")
async def chat(body: ChatIn):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Empty message")
    sid = body.session_id[:64]
    history = sessions.setdefault(sid, [])
    sessions.move_to_end(sid)
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)

    now = datetime.now(IST)
    system_text = SYSTEM_PROMPT.format(today=now.date().isoformat(), weekday=now.strftime("%A"))
    history.append(types.Content(role="user", parts=[types.Part(text=message)]))

    tool_log, sources = [], []
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            resp = await _generate(history, system_text)
            content = resp.candidates[0].content
            calls = [p.function_call for p in (content.parts or []) if p.function_call]
            history.append(content)
            if not calls:
                reply = (resp.text or "").strip() or "माफ़ कीजिए, मैं जवाब नहीं बना पाया। कृपया दोबारा लिखिए।"
                return {"reply": reply, "tool_calls": tool_log, "sources": sources}
            parts = []
            for fc in calls:
                args = dict(fc.args or {})
                out = await _call_tool(fc.name, args)
                tool_log.append({"name": fc.name, "args": args, "result": out[:300]})
                if fc.name == "search_knowledge_base":
                    try:
                        for r in json.loads(out).get("results", []):
                            tag = f"{r['source']} › {r['title']}"
                            if tag not in sources:
                                sources.append(tag)
                    except Exception:
                        pass
                parts.append(types.Part.from_function_response(name=fc.name, response={"result": out}))
            history.append(types.Content(role="user", parts=parts))
        return {"reply": "काम पूरा नहीं हो पाया, कृपया दोबारा कोशिश कीजिए।", "tool_calls": tool_log, "sources": sources}
    except genai_errors.APIError as e:
        print('GEMINI ERROR:', e)
        if history and history[-1].role == "user":
            history.pop()
        return {
            "reply": "अभी AI सर्विस व्यस्त है, कृपया एक मिनट बाद दोबारा कोशिश कीजिए।",
            "tool_calls": tool_log,
            "sources": sources,
        }


@app.get("/health")
async def health():
    return {"status": "ok", "tools": [d.name for d in state["declarations"]]}


@app.get("/")
async def home():
    return FileResponse(BASE / "static" / "index.html")