import asyncio
import json
import time
from typing import Optional, List, Dict, Any, AsyncIterator
from contextlib import AsyncExitStack
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

app = FastAPI(title="MCP-OpenAI Gateway", version="1.0.0")

# 全局配置
REQUIRE_TOOL_APPROVAL = False  # 是否需要工具调用确认
APPROVAL_TIMEOUT = 30  # 确认超时时间（秒）

# 存储待确认的工具调用
pending_approvals: Dict[str, Dict[str, Any]] = {}


# ============ Pydantic Models (OpenAI Compatible) ============

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None
    require_tool_approval: Optional[bool] = None  # 是否需要工具确认


class ToolApprovalRequest(BaseModel):
    approval_id: str
    approved: bool
    reason: Optional[str] = None  # 拒绝原因


class ToolApprovalResponse(BaseModel):
    approval_id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: str  # "pending", "approved", "rejected", "timeout"
    timestamp: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# ============ MCP Client Manager ============

class MCPClientManager:
    """管理 MCP 连接和工具"""
    
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict[str, Any]] = []
        self.mcp_server_url: Optional[str] = None
        
    async def connect(self, server_url: str):
        """连接到 MCP 服务器"""
        if self.session:
            return  # 已连接
            
        self.mcp_server_url = server_url
        streams = await self.exit_stack.enter_async_context(
            sse_client(url=server_url)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*streams)
        )
        await self.session.initialize()
        
        # 获取工具列表
        response = await self.session.list_tools()
        self.tools = self._format_tools(response.tools)
        print(f"✅ 已连接到 MCP 服务器: {server_url}")
        print(f"🔧 加载了 {len(self.tools)} 个工具")
        
    def _format_tools(self, mcp_tools) -> List[Dict[str, Any]]:
        """将 MCP 工具格式转换为 OpenAI 工具格式"""
        formatted = []
        for tool in mcp_tools:
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
            formatted.append(tool_dict)
        return formatted
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], 
                       require_approval: bool = False, 
                       approval_id: str = None) -> Dict[str, Any]:
        """调用 MCP 工具"""
        if not self.session:
            raise HTTPException(status_code=500, detail="MCP 未连接")
        
        # 如果需要确认且还没有确认
        if require_approval and approval_id:
            # 等待用户确认
            approval = await self._wait_for_approval(approval_id)
            if not approval["approved"]:
                return {
                    "status": "error", 
                    "message": f"用户拒绝执行: {approval.get('reason', '无原因')}"
                }
            
        try:
            response = await self.session.call_tool(tool_name, arguments)
            if response.isError:
                error_msg = response.content[0].text if response.content else "未知错误"
                return {"status": "error", "message": error_msg}
            else:
                result = response.content[0].text if response.content else ""
                return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _wait_for_approval(self, approval_id: str, timeout: int = APPROVAL_TIMEOUT) -> Dict[str, Any]:
        """等待用户确认"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if approval_id in pending_approvals:
                approval = pending_approvals[approval_id]
                if approval["status"] in ["approved", "rejected"]:
                    return approval
            await asyncio.sleep(0.5)
        
        # 超时
        if approval_id in pending_approvals:
            pending_approvals[approval_id]["status"] = "timeout"
        return {"approved": False, "reason": "确认超时"}
    
    async def cleanup(self):
        """清理连接"""
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"清理时出错: {e}")


# 全局 MCP 客户端实例
mcp_manager = MCPClientManager()

# OpenAI 客户端配置
openai_client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-fvysgovrkoqorpbqpubqfbvfkwfwqkpthtmvzsvdwfkmwpzz"
)


# ============ API Endpoints ============

@app.on_event("startup")
async def startup_event():
    """启动时连接 MCP 服务器"""
    # 从环境变量或配置文件读取
    import os
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9099/sse")
    try:
        await mcp_manager.connect(mcp_url)
    except Exception as e:
        print(f"⚠️ 警告: 无法连接到 MCP 服务器: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理连接"""
    await mcp_manager.cleanup()


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen3-8B",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "system"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的聊天补全接口"""
    
    # 确定是否需要工具确认
    require_approval = request.require_tool_approval
    if require_approval is None:
        require_approval = REQUIRE_TOOL_APPROVAL
    
    # 转换消息格式
    messages = [msg.dict(exclude_none=True) for msg in request.messages]
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, messages, require_approval),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_chat_completion(request, messages, require_approval)


async def non_stream_chat_completion(
    request: ChatCompletionRequest, 
    messages: List[Dict[str, Any]],
    require_approval: bool = False
) -> ChatCompletionResponse:
    """非流式响应"""
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    # 处理工具调用循环
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 [非流式] 第 {iteration} 轮对话开始...")
        
        response = openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            tools=mcp_manager.tools if mcp_manager.tools else None,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        choice = response.choices[0]
        assistant_message = choice.message
        
        # 如果没有工具调用，返回结果
        if not assistant_message.tool_calls:
            print(f"✅ [非流式] 对话完成，无工具调用")
            print(f"💬 回复: {assistant_message.content[:100]}...")
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=assistant_message.content
                        ),
                        finish_reason=choice.finish_reason
                    )
                ],
                usage=Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            )
        
        # 处理工具调用
        print(f"🔧 [非流式] 检测到 {len(assistant_message.tool_calls)} 个工具调用")
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in assistant_message.tool_calls
            ]
        })
        
        # 执行所有工具
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"  ⚙️ 调用工具: {tool_name}")
            print(f"  📥 参数: {json.dumps(tool_args, ensure_ascii=False)}")
            
            # 如果需要确认
            approval_id = None
            if require_approval:
                approval_id = f"approval-{uuid.uuid4().hex[:8]}"
                pending_approvals[approval_id] = {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "status": "pending",
                    "timestamp": int(time.time()),
                    "approved": False
                }
                print(f"  ⏳ 等待用户确认 (ID: {approval_id})...")
            
            result = await mcp_manager.call_tool(
                tool_name, 
                tool_args, 
                require_approval=require_approval,
                approval_id=approval_id
            )
            print(f"  ✅ 结果: {json.dumps(result, ensure_ascii=False)[:100]}...")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False)
            })
    
    # 超过最大迭代次数
    raise HTTPException(status_code=500, detail="达到最大工具调用次数")


async def stream_chat_completion(
    request: ChatCompletionRequest,
    messages: List[Dict[str, Any]],
    require_approval: bool = False
) -> AsyncIterator[str]:
    """流式响应"""
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 [流式] 第 {iteration} 轮对话开始...")
        
        stream = openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            tools=mcp_manager.tools if mcp_manager.tools else None,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        )
        
        full_content = ""
        tool_calls = {}
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            # 流式输出内容
            if delta.content:
                full_content += delta.content
                print(delta.content, end="", flush=True)  # 服务端打印
                chunk_data = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content=delta.content),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk_data.json()}\n\n"
            
            # 收集工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = tc
                    else:
                        if tc.function.arguments:
                            tool_calls[idx].function.arguments += tc.function.arguments
        
        # 如果没有工具调用，结束
        if not tool_calls:
            print(f"\n✅ [流式] 对话完成，无工具调用")
            chunk_data = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {chunk_data.json()}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # 处理工具调用
        tool_calls_list = list(tool_calls.values())
        print(f"\n🔧 [流式] 检测到 {len(tool_calls_list)} 个工具调用")
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls_list
            ]
        })
        
        for tool_call in tool_calls_list:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"  ⚙️ 调用工具: {tool_name}")
            print(f"  📥 参数: {json.dumps(tool_args, ensure_ascii=False)}")
            
            # 如果需要确认
            approval_id = None
            if require_approval:
                approval_id = f"approval-{uuid.uuid4().hex[:8]}"
                pending_approvals[approval_id] = {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "status": "pending",
                    "timestamp": int(time.time()),
                    "approved": False
                }
                print(f"  ⏳ 等待用户确认 (ID: {approval_id})...")
            
            result = await mcp_manager.call_tool(
                tool_name, 
                tool_args,
                require_approval=require_approval,
                approval_id=approval_id
            )
            print(f"  ✅ 结果: {json.dumps(result, ensure_ascii=False)[:100]}...")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False)
            })


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "mcp_connected": mcp_manager.session is not None,
        "tools_count": len(mcp_manager.tools),
        "require_approval": REQUIRE_TOOL_APPROVAL,
        "pending_approvals": len([a for a in pending_approvals.values() if a["status"] == "pending"])
    }


@app.get("/v1/tool_approvals")
async def list_pending_approvals():
    """列出所有待确认的工具调用"""
    pending = {
        aid: {
            "approval_id": aid,
            "tool_name": info["tool_name"],
            "arguments": info["arguments"],
            "status": info["status"],
            "timestamp": info["timestamp"]
        }
        for aid, info in pending_approvals.items()
        if info["status"] == "pending"
    }
    return {"pending_approvals": list(pending.values())}


@app.post("/v1/tool_approvals/{approval_id}")
async def approve_tool_call(approval_id: str, request: ToolApprovalRequest):
    """确认或拒绝工具调用"""
    if approval_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="确认ID不存在")
    
    approval = pending_approvals[approval_id]
    if approval["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"该确认已处理: {approval['status']}")
    
    # 更新状态
    approval["status"] = "approved" if request.approved else "rejected"
    approval["approved"] = request.approved
    if request.reason:
        approval["reason"] = request.reason
    
    print(f"{'✅ 用户批准' if request.approved else '❌ 用户拒绝'} 工具调用: {approval['tool_name']} (ID: {approval_id})")
    
    return {
        "approval_id": approval_id,
        "status": approval["status"],
        "tool_name": approval["tool_name"]
    }


@app.get("/v1/config")
async def get_config():
    """获取配置"""
    return {
        "require_tool_approval": REQUIRE_TOOL_APPROVAL,
        "approval_timeout": APPROVAL_TIMEOUT,
        "mcp_server_url": mcp_manager.mcp_server_url
    }


@app.post("/v1/config")
async def update_config(require_approval: bool = None, timeout: int = None):
    """更新配置"""
    global REQUIRE_TOOL_APPROVAL, APPROVAL_TIMEOUT
    
    if require_approval is not None:
        REQUIRE_TOOL_APPROVAL = require_approval
    if timeout is not None:
        APPROVAL_TIMEOUT = timeout
    
    return await get_config()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)