import os
import json
import requests
import cmd
import openai
from datetime import datetime
from typing import Any, List, Dict


# ========== 工具函数 ==========
def load_user_data(user_id: str) -> Dict:
    path = f"./history/{user_id}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sessions": []}


def save_user_data(user_id: str, data: Dict):
    os.makedirs("history", exist_ok=True)
    path = f"./history/{user_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ========== CLI 主体 ==========
class ChatCLI(cmd.Cmd):
    intro = "🤖 欢迎使用 ChatCLI — 输入 help 或 ? 查看命令。\n"
    prompt = "(chat-cli) "

    def __init__(self):
        super().__init__()
        self.server_url = "http://127.0.0.1:8000"
        self.client = openai.OpenAI(base_url=f"{self.server_url}/v1", api_key="dummy")
        self.user_id = None
        self.user_data: Dict = {"sessions": []}
        self.current_session: Dict = None
        self.stream_mode = True
        self.model = "Qwen/Qwen3-8B"

    def do_help(self, arg):
        """显示命令帮助"""
        if arg:
            # 显示单个命令的帮助
            cmd = getattr(self, f"help_{arg}", None)
            if cmd:
                cmd()
            else:
                func = getattr(self, f"do_{arg}", None)
                if func and func.__doc__:
                    print(func.__doc__)
                else:
                    print(f"未找到命令 '{arg}' 的帮助。")
        else:
            print("\n💬 ChatCLI 可用命令：\n")
            print("  login       登录到 OpenAI 或兼容模型服务")
            print("  chat        与模型进行对话")
            print("  tools       查看或配置 MCP 工具连接")
            print("  sessions    查看或切换聊天会话")
            print("  health      测试后端健康状态")
            print("  quit        退出程序")
            print("\n输入 `help <命令名>` 查看详细说明，例如：help chat\n")


    # ========== 登录 ==========
    def do_login(self, arg):
        """登录账户"""
        username = input("👤 用户名: ").strip()
        password = input("🔑 密码: ").strip()
        try:
            res = requests.post(
                f"{self.server_url}/login",
                json={"user_id": username, "password": password}
            )
            if res.status_code == 200:
                self.user_id = username
                self.user_data = load_user_data(username)
                self.prompt = f"(chat-cli:{self.user_id}) "
                print(f"✅ 登录成功，已加载 {len(self.user_data['sessions'])} 个会话。")
                print("💡 输入 sessions 查看所有会话，输入 chat 开始新对话。")
            else:
                print("❌ 登录失败：用户名或密码错误。")
        except Exception as e:
            print(f"⚠️ 登录出错：{e}")

    # ========== 查看服务器状态 ==========
    def do_health(self, arg):
        """检查服务器健康状态"""
        try:
            res = requests.get(f"{self.server_url}/health")
            print(json.dumps(res.json(), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"⚠️ 服务器健康检查失败：{e}")

    # ========== 列出所有会话 ==========
    def do_sessions(self, arg):
        """列出所有会话"""
        if not self.user_id:
            print("⚠️ 请先登录。")
            return
        sessions = self.user_data.get("sessions", [])
        if not sessions:
            print("📭 暂无会话记录。")
            return
        print("\n🗂 历史会话列表：")
        for i, s in enumerate(sessions):
            title = s.get("title", f"会话{i+1}")
            print(f"{i+1}. {title} ({len(s.get('messages', []))} 条消息)")
        while True:
            choice = input("\n💡 输入编号选择会话，或输入q退出: ").strip()
            if choice == "q":
                print("📭 已取消选择。")
                return
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(sessions):
                    print("❌ 无效编号，请重新输入。")
                    continue
                self.current_session = sessions[idx]
                self.messages = self.current_session["messages"]
                print(f"✅ 已切换到会话 {idx+1}: {self.current_session.get('title','无标题')}")
                self.do_chat("")  # 进入对话
                break
            except ValueError:
                print("❌ 请输入有效的数字编号。")

    # ========== 新建或继续会话 ==========
    def do_chat(self, arg):
        """开始一个新对话（或继续当前会话）"""
        if not self.user_id:
            print("⚠️ 请先登录。")
            return

        # 如果当前没有选定会话，就创建新会话
        if self.current_session is None:
            session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.current_session = {
                "id": session_id,
                "title": f"新对话_{session_id}",
                "messages": []
            }
            self.user_data["sessions"].append(self.current_session)
            print(f"\n🆕 新建会话: {self.current_session['title']}")
        else:
            print(f"\n💬 继续会话: {self.current_session.get('title', self.current_session['id'])}")

        # 将当前会话的消息绑定到 self.messages
        self.messages = self.current_session["messages"]

        print("\n=== 进入对话模式 ===")
        print("💡 输入 '/exit' 退出，'/stream' 切换流式，'/normal' 切换普通。")

        while True:
            user_input = input("\n🧑 你: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit"]:
                save_user_data(self.user_id, self.user_data)
                print(f"💾 已保存到 history/{self.user_id}.json")
                print("👋 退出对话模式。")
                break

            if user_input == "/stream":
                self.stream_mode = True
                print("✅ 已切换到流式模式。")
                continue

            if user_input == "/normal":
                self.stream_mode = False
                print("✅ 已切换到非流式模式。")
                continue

            self.messages.append({"role": "user", "content": user_input})
            if self.stream_mode:
                self._stream_chat()
            else:
                self._complete_chat()

    # ========== 流式对话 ==========
    def _stream_chat(self):
        print("\n🤖 AI (流式): ", end="", flush=True)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True
            )
            response_text = ""
            reasoning_text = ""
            tool_calls = {}
            tool_results = {}
            for chunk in stream:
                delta = chunk.choices[0].delta
                if getattr(delta, "reasoning_content", None):
                    print(delta.reasoning_content, end="", flush=True)
                    reasoning_text += delta.reasoning_content
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        if tc.function and tc.function.name:
                            print(f"\n🛠 调用函数: {tc.function.name}")
                        if tc.function and tc.function.arguments:
                            print(tc.function.arguments, end="", flush=True)
                if getattr(delta, "role", None) == "tool":
                    tool_call_id = getattr(delta, "tool_call_id", None)
                    tool_name = getattr(delta, "name", None)
                    tool_content = getattr(delta, "content", None)
                    print(f"\n📦 工具返回结果({tool_name}, id={tool_call_id}): {tool_content}")
                    tool_results[tool_call_id] = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_content
                    }
                elif delta.content:
                    print(delta.content, end="", flush=True)
                    response_text += delta.content

            assistant_msg = {"role": "assistant", "content": response_text}
            if reasoning_text:
                assistant_msg["reasoning_content"] = reasoning_text

            #TODO: 到底应不应该保留tools执行的结果
            for tool_result in tool_results.values():
                self.messages.append(tool_result)
            self.messages.append(assistant_msg)
        except Exception as e:
            print(f"\n❌ 错误：{e}")

    # ========== 非流式对话 ==========
    def _complete_chat(self):
        print("\n🤖 AI (非流式):")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            msg = response.choices[0].message
            print(msg.content)
            self.messages.append({"role": "assistant", "content": msg.content})
        except Exception as e:
            print(f"❌ 错误：{e}")

    # ========== 退出 ==========
    def do_quit(self, arg):
        """退出程序"""
        if self.user_id:
            save_user_data(self.user_id, self.user_data)
            print(f"💾 已保存会话到 history/{self.user_id}.json")
        print("👋 Goodbye!")
        return True

    # ========== MCP 工具管理 ==========
    def do_tools(self, arg):
        """
        查看或配置 MCP 工具:
        tools              - 查看当前工具列表或管理连接
        """
        try:
            res = requests.get(f"{self.server_url}/mcp/tools")
            if res.status_code == 200:
                tools = res.json().get("tools", [])
                if tools:
                    print(f"\n🔧 已加载 {len(tools)} 个工具:")
                    for i, t in enumerate(tools):
                        fn = t["function"]
                        print(f"{i+1}. {fn['name']}: {fn.get('description', '')}")
                else:
                    print("📭 当前没有 MCP 工具加载。")
                # 提示是否更换连接
                choice = input("\n💡 是否要更换 MCP 连接？(y/N): ").strip().lower()
                if choice == "y":
                    url = input("🔗 输入 MCP URL 以连接: ").strip()
                    if not url:
                        print("❌ URL 为空，取消连接。")
                        return
                    self._connect_mcp(url)
            else:
                print("❌ 获取工具列表失败。")
        except Exception:
            print("📭 尚未连接 MCP。")
            url = input("🔗 输入 MCP URL 以连接: ").strip()
            if url:
                self._connect_mcp(url)


    def _connect_mcp(self, url: str):
        try:
            print(f"🔗 正在连接 MCP: {url}")
            res = requests.post(f"{self.server_url}/mcp/connect", json={"url": url})
            if res.status_code == 200:
                tools = res.json().get("tools", [])
                print(f"✅ 连接成功，加载 {len(tools)} 个工具。")
            else:
                print(f"❌ 连接失败: {res.text}")
        except Exception as e:
            print(f"⚠️ MCP 连接出错：{e}")

if __name__ == "__main__":
    ChatCLI().cmdloop()
