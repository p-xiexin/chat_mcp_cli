import asyncio
import json
import time
import uuid
from typing import List, Dict, Any, AsyncIterator, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import AsyncExitStack, asynccontextmanager
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

# ===================== MCP 客户端封装 =====================

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict[str, Any]] = []

    async def connect(self, url: str):
        if self.session:
            return
        streams = await self.exit_stack.enter_async_context(sse_client(url=url))
        self.session = await self.exit_stack.enter_async_context(ClientSession(*streams))
        await self.session.initialize()
        response = await self.session.list_tools()
        self.tools = self._format_tools(response.tools)
        print(f"✅ MCP connected: {len(self.tools)} tools loaded.")

    def _format_tools(self, mcp_tools) -> List[Dict[str, Any]]:
        tools = []
        for t in mcp_tools:
            schema = t.inputSchema
            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema
                }
            })
        return tools

    async def call_tool(self, name: str, args: Dict[str, Any]) -> str:
        if not self.session:
            raise HTTPException(500, "MCP not connected")
        resp = await self.session.call_tool(name, args)
        if resp.isError:
            msg = resp.content[0].text if resp.content else "Unknown MCP error"
            return json.dumps({"error": msg}, ensure_ascii=False)
        return resp.content[0].text if resp.content else "{}"

    async def close(self):
        await self.exit_stack.aclose()

mcp = MCPClient()

# ===================== 业务逻辑 =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI 启动")
    yield
    print("🛑 FastAPI 关闭")
    await mcp.close()

app = FastAPI(title="MCP-OpenAI Gateway", version="2.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "mcp_connected": bool(mcp.session), "tools": len(mcp.tools)}

# ===================== Login =====================
class LoginRequest(BaseModel):
    user_id: str
    password: str
@app.post("/login")
async def login(req: LoginRequest):
    if req.user_id == "admin" and req.password == "admin":
        return {"message": f"User {req.user_id} logged in successfully."}
    elif req.user_id == "pxx" and req.password == "pxx":
        return {"message": f"User {req.user_id} logged in successfully."}
    raise HTTPException(401, "Invalid credentials")

# ===================== MCP Config =====================
class MCPConnectReq(BaseModel):
    url: str
@app.post("/mcp/connect")
async def connect_mcp(req: MCPConnectReq):
    try:
        await mcp.connect(req.url)
        return {"status": "ok", "tools": mcp.tools}
    except Exception as e:
        raise HTTPException(500, f"MCP连接失败: {e}")

@app.get("/mcp/tools")
async def list_tools():
    return {"tools": mcp.tools}


# ===================== Chat Completions =====================
# https://platform.openai.com/docs/guides/streaming-responses?api-mode=responses
# https://platform.openai.com/docs/api-reference/responses-streaming/response/created
@app.post("/v1/chat/completions")
async def chat_completions(request: Dict[str, Any]):
    """仿 OpenAI Responses API 的统一入口"""
    if request.get("stream"):
        return StreamingResponse(
            stream_chat(request),
            media_type="text/event-stream"
        )
    else:
        return await complete_chat(request)


import json
from fastapi import HTTPException

async def complete_chat(req: Dict[str, Any]):
    """非流式推理：保持兼容 OpenAI SDK 返回结构，并在返回体上附加 `trace` 字段。"""
    messages = req["messages"]
    model = req["model"]

    trace = []

    for step in range(10):
        #TODO: model是平台二次存储的模型名+地址
        client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-fvysgovrkoqorpbqpubqfbvfkwfwqkpthtmvzsvdwfkmwpzz"
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=mcp.tools or None,
            temperature=req.get("temperature", 1.0),
            max_tokens=req.get("max_tokens"),
        )

        # 把 SDK 对象转成字典（安全可序列化）
        resp_dict = response.model_dump()
        choice = resp_dict["choices"][0]
        message = choice.get("message", {})

        # 记录本轮 assistant 的输出（使用 dict，避免非序列化对象）
        trace.append({
            "type": "assistant",
            "step": step,
            "message": message,
            "tool_calls": message.get("tool_calls")
        })

        if not message.get("tool_calls"):
            resp_dict["trace"] = trace
            return resp_dict

        for tc in message["tool_calls"]:
            raw_args = tc.get("function", {}).get("arguments", "")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception as e:
                parsed_args = {}
                tool_result = {"status": "error", "message": f"parse arguments failed: {e}"}
            else:
                tool_result = await mcp.call_tool(tc["function"]["name"], parsed_args)

            # 记录工具返回
            trace.append({
                "type": "tool_result",
                "step": step,
                "tool_call_id": tc.get("id"),
                "tool_name": tc.get("function", {}).get("name"),
                "arguments": parsed_args,
                "result": tool_result,
            })

            # 把工具结果添加回 messages（让下一轮模型看到）
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "name": tc.get("function", {}).get("name"),
                "content": json.dumps(tool_result, ensure_ascii=False) if not isinstance(tool_result, str) else tool_result
            })

        # 把 assistant 的 tool_call 占位消息也加入上下文（与模型对话格式一致）
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": message["tool_calls"]
        })

    raise HTTPException(status_code=500, detail="Reached max tool call iterations")


async def stream_chat(req: Dict[str, Any]) -> AsyncIterator[str]:
    """流式版本"""
    messages = req["messages"]
    model = req["model"]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    for i in range(10):
        client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-fvysgovrkoqorpbqpubqfbvfkwfwqkpthtmvzsvdwfkmwpzz"
        )
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=mcp.tools or None,
            temperature=req.get("temperature", 1.0),
            stream=True
        )

        tool_calls: Dict[int, Any] = {}
        for chunk in stream:
            delta = chunk.choices[0].delta
            yield f"data: {chunk.model_dump_json()}\n\n"

            # 收集工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = tc
                    else:
                        if tc.function.arguments:
                            tool_calls[idx].function.arguments += tc.function.arguments
        if not tool_calls:
            yield "data: [DONE]\n\n"
            return
        else:
            messages.append({
                "role": "assistant",
                "tool_calls": [tc.dict() for tc in tool_calls.values()]
            })

        # 工具调用
        for tc in tool_calls.values():
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                print("⚠️ 工具调用参数不完整:", tc.function.arguments)
                args = {}
            result = await mcp.call_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result
            })

            # === 以 OpenAI 流格式发送工具结果 ===
            # https://platform.openai.com/docs/guides/function-calling#streaming
            tool_event = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": i,
                        "delta": {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": result,
                        },
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(tool_event, ensure_ascii=False)}\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)