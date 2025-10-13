import asyncio
from contextlib import AsyncExitStack
import json
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI
import requests


class MCPClient:
    """负责管理与 MCP Server 的连接"""

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

    async def connect(self, server_url: str) -> ClientSession:
        """连接到 SSE 服务器并返回 session"""
        streams = await self.exit_stack.enter_async_context(sse_client(url=server_url))
        self.session = await self.exit_stack.enter_async_context(ClientSession(*streams))
        await self.session.initialize()
        return self.session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表并格式化"""
        response = await self.session.list_tools()
        tools = response.tools
        formatted_tools = []
        for tool in tools:
            properties = {
                param_name: {
                    "type": param_info["type"],
                    "description": param_info.get("description", param_info.get("title", ""))
                }
                for param_name, param_info in tool.inputSchema.get("properties", {}).items()
            }
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": tool.inputSchema.get("required", [])
                    }
                }
            }
            formatted_tools.append(tool_dict)
        return formatted_tools

    async def cleanup(self):
        """关闭连接"""
        try:
            await self.exit_stack.aclose()
        except asyncio.CancelledError:
            pass


class ChatSession:
    """负责聊天逻辑与工具调用"""

    def __init__(self, model: str, tools: List[Dict[str, Any]], session: ClientSession):
        self.model = model
        self.tools = tools
        self.session = session
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "你是一个有用的助手，可以帮助用户查找和了解实时信息。"}
        ]
        self.openai_client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-fvysgovrkoqorpbqpubqfbvfkwfwqkpthtmvzsvdwfkmwpzz"  # ⚠️ 请替换为实际的 API Key
        )

    async def process_tool_calls(self, tool_calls):
        """处理工具调用并更新消息历史"""
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in tool_calls
            ]
        })

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            try:
                tool_response = await self.session.call_tool(tool_name, tool_args)
                if tool_response.isError:
                    error_message = tool_response.content[0].text if tool_response.content else "未知错误"
                    formatted = {"status": "error", "message": error_message}
                else:
                    result = tool_response.content[0].text if tool_response.content else ""
                    formatted = {"status": "success", "result": result}

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(formatted, ensure_ascii=False)
                })

            except Exception as e:
                formatted_error = {"status": "error", "message": str(e)}
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(formatted_error, ensure_ascii=False)
                })

    async def chat(self, user_input: str) -> str:
        """处理聊天消息"""
        self.messages.append({"role": "user", "content": user_input})
        done = False

        while not done:
            stream = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                stream=True,
            )

            full_response = ""
            final_tool_calls = {}

            for chunk in stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    full_response += delta.content

                for tool_call in delta.tool_calls or []:
                    idx = tool_call.index
                    if idx not in final_tool_calls:
                        final_tool_calls[idx] = tool_call
                    else:
                        final_tool_calls[idx].function.arguments += tool_call.function.arguments

            if final_tool_calls:
                await self.process_tool_calls(list(final_tool_calls.values()))
                continue
            else:
                self.messages.append({"role": "assistant", "content": full_response})
                done = True

        return full_response




app = FastAPI()

# 用于维护所有用户的聊天会话
user_sessions: Dict[str, ChatSession] = {}
online_users: Dict[str, str] = {}  # 记录在线用户

class RequestBody(BaseModel):
    server_url: str
    user_input: str
@app.post("/openai/completion/{user_id}")
async def openai_completion(user_id: str, request: RequestBody):
    """为每个用户提供聊天和工具调用服务"""
    if user_id not in online_users:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户未登录")

    try:
        # 初始化 MCP 客户端并连接
        client = MCPClient()
        session = await client.connect(request.server_url)
        tools = await client.list_tools()
        print("mcp ok")

        # 检查该用户是否已有会话，若没有则创建新的会话
        if user_id not in user_sessions:
            chat_session = ChatSession("Qwen/Qwen3-8B", tools, session)
            user_sessions[user_id] = chat_session
        else:
            chat_session = user_sessions[user_id]
        print("chatsession ok")

        # 处理用户输入
        response = await chat_session.chat(request.user_input)

        # 清理 MCP 客户端连接
        await client.cleanup()

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")


@app.post("/openai/close/{user_id}")
async def close_session(user_id: str):
    """结束用户会话并清理历史记录"""
    if user_id in user_sessions:
        del user_sessions[user_id]
        del online_users[user_id]
        return {"message": f"Session for user {user_id} closed."}
    else:
        raise HTTPException(status_code=404, detail="User session not found.")

@app.get("/openai/history/{user_id}")
async def get_history(user_id: str):
    """获取用户的历史聊天记录"""
    if user_id in user_sessions:
        chat_session = user_sessions[user_id]
        return {"history": chat_session.messages}
    else:
        raise HTTPException(status_code=404, detail="User session not found.")

@app.get("/health")
async def health_check():
    """检查API健康状态"""
    return {"status": "ok"}

class LoginRequest(BaseModel):
    user_id: str
    password: str
@app.post("/login")
async def login(request: LoginRequest):
    """用户登录"""
    if request.user_id == "admin" and request.password == "admin":
        online_users[request.user_id] = "admin"
        return {"message": f"User {request.user_id} logged in successfully."}
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

@app.get("/online_users")
async def get_online_users():
    """获取在线用户列表"""
    return {"online_users": list(online_users.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
