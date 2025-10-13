import asyncio
import json
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from datetime import datetime

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI


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
        print("✅ 已初始化 SSE 客户端")
        return self.session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表并格式化"""
        response = await self.session.list_tools()
        tools = response.tools
        print("🔧 MCP Server 提供的工具：")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

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

        print("\n📦 转换后的工具格式：")
        print(json.dumps(formatted_tools, indent=2, ensure_ascii=False))
        return formatted_tools

    async def cleanup(self):
        """关闭连接"""
        try:
            await self.exit_stack.aclose()
        except asyncio.CancelledError:
            # 忽略，因为是正常退出
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
            api_key="sk-fvysgovrkoqorpbqpubqfbvfkwfwqkpthtmvzsvdwfkmwpzz"  # ⚠️ 不要硬编码，改用环境变量
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
            print(f"\n⚙️ 模型请求调用工具: {tool_call.function.name,} 参数: {tool_call.function.arguments}")
            confirm = input("是否执行该工具? [Y/n]: ").strip().lower()                        
            # print(f"🔧 调用工具: {tool_name}")
            # print(f"📥 参数: {json.dumps(tool_args, indent=2, ensure_ascii=False)}")
            if confirm in ("y", "yes", ""):
                pass
            else:
                print("❌ 工具调用已取消。")
                # 可选：把拒绝信息反馈给模型
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": '{"error": "user_denied", "message": "用户拒绝执行该工具"}'
                })
                continue

            try:
                tool_response = await self.session.call_tool(tool_name, tool_args)
                if tool_response.isError:
                    error_message = tool_response.content[0].text if tool_response.content else "未知错误"
                    print(f"❌ 工具执行错误: {error_message}")
                    formatted = {"status": "error", "message": error_message}
                else:
                    result = tool_response.content[0].text if tool_response.content else ""
                    print(f"✅ 工具执行成功:\n {result}")
                    formatted = {"status": "success", "result": result}

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(formatted, ensure_ascii=False)
                })

            except Exception as e:
                print(f"⚠️ 工具调用异常: {e}")
                formatted_error = {"status": "error", "message": str(e)}
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(formatted_error, ensure_ascii=False)
                })

    async def chat_loop(self):
        """主对话循环"""
        print("🤖 欢迎使用 AI 助手！(按 Ctrl+C 退出)")
        try:
            while True:
                user_input = input("\n用户: ").strip()
                if not user_input:
                    continue
                self.messages.append({"role": "user", "content": user_input})

                print("\nAI: ", end="", flush=True)
                done = False

                # --- 工具触发规则 ---
                if user_input.startswith("/rag "):
                    text = user_input.replace("/rag ", "")
                    print("正在检索相关文献")
                    # await self.handle_tool_calls([{
                    #     "id": "local-save",
                    #     "name": "string_to_file",
                    #     "arguments": {"text": text}
                    # }])

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

                        # 正式输出
                        if delta.content:
                            full_response += delta.content
                            print(delta.content, end="", flush=True)

                        # 思维链（reasoning）
                        # 正常不需要保存 reasoning_content 到 messages，只在需要时作为调试/展示日志保存。
                        if getattr(delta, "reasoning_content", None):
                            print(delta.reasoning_content, end="", flush=True)

                        # summary 捕获
                        if getattr(delta, "summary", None):
                            print(delta.summary, flush=True)

                        # 工具调用拼接
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

        except KeyboardInterrupt:
            print("\n👋 再见！")
            self._save_session()

    def _save_session(self):
        """保存对话历史到 JSON 文件"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"chat_history_{ts}.json"
        filename = f"chat_history.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        print(f"💾 已保存对话历史到 {filename}")


async def main():
    if len(sys.argv) < 2:
        print("使用方法: uv run client.py <SSE MCP服务器的URL>")
        sys.exit(1)

    client = MCPClient()
    try:
        session = await client.connect(sys.argv[1])
        tools = await client.list_tools()
        chat = ChatSession("Qwen/Qwen3-8B", tools, session)
        await chat.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
