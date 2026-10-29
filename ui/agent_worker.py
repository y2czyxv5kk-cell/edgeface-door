# -*- coding: utf-8 -*-
"""大模型聊天线程：LLM 推理放到后台，避免阻塞 UI。

EdgeAgent 由页面持有并跨消息复用（llama.cpp 子进程/模型只加载
一次）；本线程只负责把 chat_stream 的事件转发成 Qt 信号：
  chunk(str)   流式文本块，UI 追加到气泡
  replace(str) 工具已执行，UI 用该文本替换气泡内容
  reply(str)   完整最终文本（写历史、恢复输入用）
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.path.join(_PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from PyQt5.QtCore import QThread, pyqtSignal  # noqa: E402


class AgentWorker(QThread):
    """处理一条用户消息：流式 LLM 推理 + 工具调用。"""

    chunk = pyqtSignal(str)
    replace = pyqtSignal(str)
    reply = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, agent, user_text: str, history: list, parent=None):
        """Args:
            agent: 页面持有的共享 EdgeAgent（复用后端资源）
            user_text: 用户这条消息
            history: 之前对话历史（OpenAI 格式，不含 system）
        """
        super().__init__(parent)
        self._agent = agent
        self._text = user_text
        self._history = history

    def run(self) -> None:  # noqa: N802（QThread 约定命名）
        try:
            parts = []
            for event, value in self._agent.chat_stream(self._text, self._history):
                if event == "replace":
                    parts = [value]
                    self.replace.emit(value)
                else:
                    parts.append(value)
                    self.chunk.emit(value)
            self.reply.emit("".join(parts))
        except Exception as e:
            self.failed.emit(str(e))
