"""本地大模型层：统一数据结构 + ollama 后端。

OllamaClient 与 llama_backend.LlamaCppClient 接口一致（chat 返回
ChatReply），agent 按 [ollama → llama.cpp] 顺序容错。
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List

import httpx

import ollama


@dataclass
class ToolCall:
    """LLM 请求的一次工具调用。"""
    name: str                        # 工具名（与 ToolRegistry 注册一致）
    arguments: Dict = field(default_factory=dict)  # 参数字典
    id: str = ""                     # 调用 ID（OpenAI 协议用）


@dataclass
class ChatReply:
    """LLM 的一轮回复。

    tool_calls 非空表示模型请求执行工具；否则 content 为最终答复。
    """
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


def ollama_healthy(timeout: float = 2.0) -> bool:
    """探测 ollama 服务是否在线（2 秒超时）。"""
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


class OllamaClient:
    """ollama 后端：本地服务对话 + 工具调用。"""

    def __init__(self, model: str = "qwen2.5:1.5b", host: str = None,
                 timeout: float = 120.0, num_ctx: int = 2048):
        """Args:
            model: ollama 模型名（`ollama list` 查看）
            host: 服务地址；None 自动处理（见下）
            timeout: HTTP 超时（秒）
            num_ctx: 上下文窗口长度
        """
        # 环境变量 OLLAMA_HOST 若为 0.0.0.0（服务端监听用），
        # 客户端连接会失败，自动替换为 127.0.0.1
        resolved = host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        if "0.0.0.0" in resolved:
            resolved = resolved.replace("0.0.0.0", "127.0.0.1")

        self._client = ollama.Client(host=resolved, timeout=timeout)
        self.model = model
        self.num_ctx = num_ctx

    def _request_kwargs(self, messages: list, tools: list = None, stream: bool = False) -> dict:
        """组装请求参数（chat / chat_stream 共用）。

        temperature 压低 + num_predict 封顶：小模型对话更稳、回复不跑题拖长；
        keep_alive 让模型常驻内存，避免闲置后每轮重新加载。
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": 0.2,
                "num_predict": 256,
            },
            "keep_alive": "30m",
        }
        if tools:
            kwargs["tools"] = tools
        if stream:
            kwargs["stream"] = True
        return kwargs

    def chat(self, messages: list, tools: list = None) -> ChatReply:
        """发一轮对话（OpenAI 风格 messages + 可选工具定义）。"""
        msg = self._client.chat(**self._request_kwargs(messages, tools)).message

        tool_calls = [
            ToolCall(name=tc.function.name,
                     arguments=tc.function.arguments or {},
                     id=getattr(tc, "id", "") or "")
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        return ChatReply(content=(msg.content or "").strip(), tool_calls=tool_calls)

    def chat_stream(self, messages: list, tools: list = None):
        """流式对话：逐 chunk yield ("text", 文本)；结束时若模型请求了
        工具，再 yield ("tool_calls", [ToolCall, ...])。"""
        tool_calls = []
        for part in self._client.chat(**self._request_kwargs(messages, tools, stream=True)):
            msg = part.get("message") or {}
            content = msg.get("content") or ""
            if content:
                yield ("text", content)
            for tc in (msg.get("tool_calls") or []):
                fn = (tc.get("function") or {})
                tool_calls.append(ToolCall(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments") or {},
                    id=tc.get("id", "") or ""))
        if tool_calls:
            yield ("tool_calls", tool_calls)
