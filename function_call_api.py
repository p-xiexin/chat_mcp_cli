import asyncio
import json
import time
import uuid
from typing import List, Dict, Any, AsyncIterator, Optional
import aiohttp
from fastapi import FastAPI, HTTPException, Request, status
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
async def chat_completions(req: Request):
    body = await req.json()
    if body.get("stream"):
        async def safe_stream():
            try:
                async for chunk in stream_chat(body):
                    if await req.is_disconnected():
                        print("⚠️ 客户端断开连接，停止推送。")
                        break
                    yield chunk
            except Exception as e:
                print(f"❌ 流式异常: {e}")

        return StreamingResponse(safe_stream(), media_type="text/event-stream")

    return await complete_chat(body)


import json
from fastapi import HTTPException

def make_chunk(id, model, created, role, index=0, content=None, annotations=None, tool_call_id=None):
    delta = {"role": role}
    if content is not None: delta["content"] = content
    if annotations is not None: delta["annotations"] = annotations
    if tool_call_id: delta["tool_call_id"] = tool_call_id
    return f"data: {json.dumps({'id': id,'object':'chat.completion.chunk','created':created,'model':model,'choices':[{'index':index,'delta':delta,'finish_reason':None}]}, ensure_ascii=False)}\n\n"


# ============ 🧩 RAG 查询函数 ============ #
async def rag_retrieve(query: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    """最简版 RAG 检索函数，直接传入 query 和 extra"""
    file_search = extra.get("file_search")
    if not file_search:
        return {"results": []}

    base_url = "http://localhost:8900"   # TODO: 改为实际检索服务地址
    vector_store_ids = file_search.get("vector_store_ids", [])
    project_id = vector_store_ids[0] if vector_store_ids else "default"
    payload = {
        "query": query,
        "retrieval_mode": "hybrid",
        "top_k": file_search.get("max_num_results", 5),
        "retrieval_weight": 0.5,
    }

    url = f"{base_url}/querySimple/{project_id}"
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            data = await resp.json()
            if isinstance(data, list):
                # 返回列表时包装成字典
                results = [{"id": str(i), "filename": "", "content": str(item)} for i, item in enumerate(data)]
            else:
                results = data.get("results", [])
            return {"results": results}
        

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

    # ============ 🔍 RAG 检索阶段 ============ #
    extra = req.get("extra", {}) or {}
    file_search = extra.get("file_search")
    query = req["messages"][-1]["content"]

    if file_search:  # 只有有 file_search 才执行 RAG 查询
        try:
            rag_context = await rag_retrieve(query, extra)
            results = rag_context.get("results", [])

            if results:
                # 拼接文档内容加入系统消息
                concatenated = "\n\n".join([doc.get("content", "") for doc in results])
                req["messages"].append({
                    "role": "system",
                    "content": f"以下是与用户问题相关的参考资料：\n{concatenated}"
                })
                print("✅ 已添加RAG上下文。")

                # 流式返回每条 RAG 文档
                for i, doc in enumerate(results):
                    yield make_chunk(
                        completion_id, model, created,
                        role="file_search",
                        index=i,
                        annotations=[{
                            "id": doc.get("id"),
                            "filename": doc.get("filename"),
                            "content": doc.get("content")
                        }]
                    )
        except Exception as e:
            print(f"⚠️ RAG 检索失败: {e}")


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
            yield make_chunk(
                completion_id, model, created,
                role="tool",
                index=i,
                content=result,
                tool_call_id=tc.id
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)