"""终端聊天演示：本地大模型（ollama / llama.cpp）控制门禁。

用法：
    python tools/chat.py                              # 交互对话
    python tools/chat.py --once "帮我开一下门"         # 单轮测试
    python tools/chat.py --backend llama --once "..."  # 指定后端

可试的对话：
    帮我开一下门        → 调用 open_door
    今天有谁来过？      → 调用 recent_logs
    有哪些人能开门？    → 调用 list_users
    /quit              → 退出
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from edgebrain import EdgeAgent  # noqa: E402


def _print_stream(agent: EdgeAgent, user: str, history: list = None) -> str:
    """流式跑一轮并逐字打印；replace 事件用 \r 覆盖已打印内容。"""
    parts = []
    shown = 0
    for event, value in agent.chat_stream(user, history):
        if event == "replace":
            parts = [value]
            print("\r助手> " + value + " " * max(shown - len(value), 0),
                  end="", flush=True)
            shown = len(value)
        else:
            parts.append(value)
            print(value, end="", flush=True)
            shown += len(value)
    print()
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="端侧大模型门禁助手（CLI 演示）")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="ollama 模型名")
    parser.add_argument("--backend", default="auto", choices=["auto", "ollama", "llama"],
                        help="推理后端：auto=ollama 优先失败回落 llama.cpp")
    parser.add_argument("--once", default=None, metavar="文本",
                        help="单轮模式：处理一句话后退出")
    args = parser.parse_args()

    print("正在连接本地大模型…")
    agent = EdgeAgent(model=args.model, backend=args.backend)

    if args.once is not None:
        print(f"你> {args.once}")
        print("助手> ", end="", flush=True)
        _print_stream(agent, args.once)
        agent.close()
        return

    print("可以试试：帮我开一下门 / 今天有谁来过？/ 有哪些人能开门？")
    print("（输入 /quit 退出）\n")

    history = []  # 多轮历史（OpenAI 格式）
    while True:
        try:
            user = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "退出"):
            break

        print("助手> ", end="", flush=True)
        reply = _print_stream(agent, user, history)
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
        history = history[-10:]  # 只保留最近 10 轮
        print()

    agent.close()
    print("再见。")


if __name__ == "__main__":
    main()
