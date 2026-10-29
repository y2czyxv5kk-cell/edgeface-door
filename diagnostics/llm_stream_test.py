# -*- coding: utf-8 -*-
"""llama.cpp 后端流式验证：拉起真实 llama-server 测 SSE 流式 + 工具调用。

用法（项目根目录）：
    python diagnostics/llm_stream_test.py
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from edgebrain.llama_backend import LlamaCppClient


def main():
    t0 = time.time()
    print("拉起 llama-server（1.5B GGUF，CPU 加载需要点时间）…")
    client = LlamaCppClient()
    print(f"模型加载完成，耗时 {time.time() - t0:.1f}s")

    msgs = [{"role": "system",
             "content": "你是 EdgeFace 门禁助手。回答用中文，一句话即可。"},
            {"role": "user", "content": "帮我开一下门"}]
    tools = [{"type": "function",
              "function": {"name": "open_door", "description": "打开门禁",
                           "parameters": {"type": "object",
                                          "properties": {}, "required": []}}}]

    print("流式输出：", end="", flush=True)
    texts, tcs = [], []
    t_stream = time.time()
    for event, value in client.chat_stream(msgs, tools):
        if event == "text":
            texts.append(value)
            print(value, end="", flush=True)
        else:
            tcs.extend(value)
    print(f"\n流式耗时 {time.time() - t_stream:.1f}s")
    print("tool_calls:", [(tc.name, tc.arguments) for tc in tcs])
    print("无结构化 tool_calls 时由 agent 层规则兜底（keyword → open_door）")

    client.close()
    print("server 已关闭，进程应已退出")


if __name__ == "__main__":
    main()
