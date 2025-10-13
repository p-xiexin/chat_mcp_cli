import os
import json
import requests
import cmd
import openai
from datetime import datetime
from typing import Any, List, Dict


# ========== 工具函数 ==========
def load_history(user_id: str) -> List[Dict]:
    path = f"./history/{user_id}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(user_id: str, messages: List[Dict]):
    os.makedirs("history", exist_ok=True)
    path = f"./history/{user_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# ========== CLI 主体 ==========
class ChatCLI(cmd.Cmd):
    intro = "🤖 欢迎使用 ChatCLI — 输入 help 或 ? 查看命令。\n"
    prompt = "(chat-cli) "

    def __init__(self):
        super().__init__()
        self.server_url = "http://127.0.0.1:8000"
        self.client = openai.OpenAI(base_url=f"{self.server_url}/v1", api_key="dummy")
        self.user_id = None
        self.messages: List[Dict] = []
        self.stream_mode = True
        self.model = "Qwen/Qwen3-8B"

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
                self.messages = load_history(username)
                self.prompt = f"(chat-cli:{self.user_id}) "
                print(f"✅ 登录成功，已加载 {len(self.messages)} 条历史消息。")
            else:
                print("❌ 登录失败：用户名或密码错误。")
        except Exception as e:
            print(f"⚠️ 登录出错：{e}")

    # ========== 健康检查 ==========
    def do_health(self, arg):
        """检查服务器健康状态"""
        try:
            res = requests.get(f"{self.server_url}/health")
            print(json.dumps(res.json(), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"⚠️ 服务器健康检查失败：{e}")

    # ========== 聊天主流程 ==========
    def do_chat(self, arg):
        """开始对话，会话将自动保存"""
        # if not self.user_id:
        #     print("⚠️ 请先登录。")
        #     return

        print("\n=== 进入对话模式 ===")
        print("💡 输入 '/exit' 退出，'/stream' 切换流式，'/normal' 切换普通。")

        while True:
            user_input = input("\n🧑 你: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit"]:
                save_history(self.user_id, self.messages)
                print(f"💾 已保存到 history/{self.user_id}.json")
                print("👋 再见！")
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
                if delta.content:
                    print(delta.content, end="", flush=True)
                    response_text += delta.content
                if getattr(delta, "reasoning_content", None):
                    print(delta.reasoning_content, end="", flush=True)
                    reasoning_text += delta.reasoning_content
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        if tc.function and tc.function.name:
                            print(f"\n🛠 调用函数: {tc.function.name}\n")
                        if tc.function and tc.function.arguments:
                            print(tc.function.arguments, end="", flush=True)

            # --- 整理最终 assistant 消息 ---
            assistant_msg = {"role": "assistant", "content": response_text}
            if reasoning_text:
                assistant_msg["reasoning_content"] = reasoning_text
            if tool_calls:
                assistant_msg["tool_calls"] = list(tool_calls.values())

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

    # ========== 查看历史 ==========
    def do_history(self, arg):
        """查看历史记录"""
        if not self.user_id:
            print("⚠️ 请先登录。")
            return
        messages = load_history(self.user_id)
        if not messages:
            print("📭 暂无历史记录。")
            return
        for m in messages:
            role = "🧑" if m["role"] == "user" else "🤖"
            print(f"{role}: {m['content']}")

    # ========== 退出 ==========
    def do_quit(self, arg):
        """退出程序"""
        if self.user_id:
            save_history(self.user_id, self.messages)
            print(f"💾 已保存会话到 history/{self.user_id}.json")
        print("👋 Goodbye!")
        return True


if __name__ == "__main__":
    ChatCLI().cmdloop()
