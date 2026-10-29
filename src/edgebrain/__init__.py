"""edgebrain 包：本地大模型助手——自然语言理解 + 工具调用控制门禁。

- llm.py           统一数据结构 + ollama 后端
- llama_backend.py llama-server 二进制后端
- tools.py         工具注册表（open_door / recent_logs / list_users）
- agent.py         对话循环：LLM 决定调工具 → 执行 → 生成回复
"""
from .agent import EdgeAgent
from .tools import ToolRegistry

__all__ = ["EdgeAgent", "ToolRegistry"]
