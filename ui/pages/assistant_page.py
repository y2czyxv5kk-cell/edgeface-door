# -*- coding: utf-8 -*-
"""智能助手页：与本地大模型对话，大模型可调用工具控制门禁。

输入 → AgentWorker 后台跑 EdgeAgent（流式 LLM + 工具调用）→
chunk/replace 信号逐块回 UI 线程渲染气泡（首个 token 即上屏，
无需等整段生成）。推理期间 UI 保持响应。

EdgeAgent 页面级共享：每条消息不再重建（llama.cpp 子进程/模型
只加载一次），切离页面时统一释放。

模型默认 qwen2.5:1.5b（ollama）；低内存机器可换 qwen2.5:0.5b，
llama.cpp 后端模型文件在 assets/models/llm/。
"""
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout,
    QWidget,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .. import common  # noqa: E402
from ..common import TEXT_BTN_STYLE  # noqa: E402
from ..agent_worker import AgentWorker  # noqa: E402

MODEL = "qwen2.5:1.5b"  # ollama 模型名（llama 后端忽略此参数）

_USER_BUBBLE = """
QLabel {
    background: #0e7490; color: #ffffff;
    border-radius: 10px; padding: 10px 14px;
    font-size: 16px;
}
"""
_ASSIST_BUBBLE = """
QLabel {
    background: #131e33; color: #e2e8f0;
    border-radius: 10px; padding: 10px 14px;
    font-size: 16px;
}
"""


class AssistantPage(QWidget):
    """大模型门禁助手聊天页。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageRoot")
        self.setStyleSheet(common.PAGE_STYLE)
        self._history = []          # 多轮历史（OpenAI 格式）
        self._last_sent = ""        # 最近一条用户消息
        self._worker: AgentWorker = None
        self._agent = None          # 共享 EdgeAgent（懒创建，on_leave 释放）
        self._stream_row = None     # 流式气泡所在行
        self._stream_bubble = None  # 正在流式渲染的助手气泡
        self._build_ui()

    # ---- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addWidget(common.make_page_header(
            "智能助手", "本地大模型 · 自然语言控制门禁"))

        # 消息区：QScrollArea + 纵向气泡流
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.msg_container = QWidget()
        self.msg_container.setStyleSheet("background: transparent;")
        self.msg_layout = QVBoxLayout(self.msg_container)
        self.msg_layout.setContentsMargins(8, 8, 8, 8)
        self.msg_layout.setSpacing(10)
        self.msg_layout.addStretch(1)  # 底部弹簧：气泡靠上
        self.scroll.setWidget(self.msg_container)
        root.addWidget(self.scroll, 1)

        self._add_assist_msg(
            f"你好，我是门禁助手（本地模型 {MODEL}）。\n"
            "你可以试试：\n"
            "· 帮我开一下门\n"
            "· 今天有谁来过？\n"
            "· 有哪些人能开门？")

        # 输入区
        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        self.input = QLineEdit()
        self.input.setPlaceholderText("说点什么…（回车发送）")
        self.input.setMinimumHeight(48)
        self.input.setStyleSheet(common.INPUT_STYLE)
        self.input.returnPressed.connect(self._send)
        input_row.addWidget(self.input, 1)

        self.btn_send = QPushButton("发送")
        self.btn_send.setStyleSheet(TEXT_BTN_STYLE)
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.clicked.connect(self._send)
        input_row.addWidget(self.btn_send)
        root.addLayout(input_row)

    # ---- 气泡渲染 ----
    def _add_user_msg(self, text: str) -> None:
        """用户气泡（右对齐）。"""
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setStyleSheet(_USER_BUBBLE)
        bubble.setAlignment(Qt.AlignRight)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(bubble, 4)
        self.msg_layout.insertLayout(self.msg_layout.count() - 1, row)
        self._scroll_to_bottom()

    def _add_assist_msg(self, text: str) -> None:
        """助手气泡（左对齐）。"""
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setStyleSheet(_ASSIST_BUBBLE)
        bubble.setAlignment(Qt.AlignLeft)
        row = QHBoxLayout()
        row.addWidget(bubble, 4)
        row.addStretch(1)
        self.msg_layout.insertLayout(self.msg_layout.count() - 1, row)
        self._scroll_to_bottom()

    def _remove_last_msg(self) -> None:
        """移除最后一个气泡（用于替换"思考中"占位）。"""
        count = self.msg_layout.count()
        if count <= 1:
            return
        item = self.msg_layout.takeAt(count - 2)  # 最后一个是弹簧
        if item.layout() is not None:
            while item.layout().count():
                child = item.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        del item

    def _scroll_to_bottom(self) -> None:
        """滚动到底部。"""
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ---- 流式渲染 ----
    def _ensure_stream_bubble(self, text: str = ""):
        """确保流式气泡存在；不存在则替换"思考中…"占位并创建。"""
        if self._stream_bubble is None:
            self._remove_last_msg()
            self._stream_row = QHBoxLayout()
            self._stream_bubble = QLabel(text)
            self._stream_bubble.setWordWrap(True)
            self._stream_bubble.setStyleSheet(_ASSIST_BUBBLE)
            self._stream_bubble.setAlignment(Qt.AlignLeft)
            self._stream_row.addWidget(self._stream_bubble, 4)
            self._stream_row.addStretch(1)
            self.msg_layout.insertLayout(
                self.msg_layout.count() - 1, self._stream_row)
        return self._stream_bubble

    def _on_chunk(self, text: str) -> None:
        """流式文本块：首块替换占位，之后追加到气泡。"""
        bubble = self._ensure_stream_bubble()
        bubble.setText(bubble.text() + text)
        self._scroll_to_bottom()

    def _on_replace(self, text: str) -> None:
        """工具已执行：整泡替换为工具结果。"""
        bubble = self._ensure_stream_bubble()
        bubble.setText(text)
        self._scroll_to_bottom()

    # ---- 发送与回调 ----
    def _send(self) -> None:
        """发送消息 → 启动后台 worker。推理期间输入禁用。"""
        text = self.input.text().strip()
        if not text:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        agent = self._ensure_agent()
        if agent is None:
            self._add_assist_msg(
                "⚠️ 本地大模型后端不可用。\n"
                "提示：确认 ollama 已启动，或 llama.cpp 模型文件就位，"
                "且本机内存充足。")
            return

        self._last_sent = text
        self._add_user_msg(text)
        self.input.clear()
        self.input.setEnabled(False)
        self.btn_send.setEnabled(False)
        self._add_assist_msg("思考中…")

        self._worker = AgentWorker(agent, text, self._history, parent=self)
        self._worker.chunk.connect(self._on_chunk)
        self._worker.replace.connect(self._on_replace)
        self._worker.reply.connect(self._on_reply)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _ensure_agent(self):
        """懒创建共享 EdgeAgent（首次发送时）；后端都不可用返回 None。"""
        if self._agent is None:
            try:
                from edgebrain import EdgeAgent
                self._agent = EdgeAgent(model=MODEL)
            except Exception:
                self._agent = None
        return self._agent

    def _on_reply(self, reply: str) -> None:
        """流式结束：补全最终文本，更新历史，恢复输入。"""
        bubble = self._ensure_stream_bubble()
        bubble.setText(reply or "（空回复，请重试）")
        self._stream_row = None
        self._stream_bubble = None

        self._history.append({"role": "user", "content": self._last_sent})
        self._history.append({"role": "assistant", "content": reply})
        self._history = self._history[-10:]  # 只保留最近 10 轮

        self.input.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.input.setFocus()

    def _on_failed(self, msg: str) -> None:
        """出错：显示错误提示，恢复输入。"""
        self._remove_last_msg()   # 流式气泡或"思考中…"占位
        self._stream_row = None
        self._stream_bubble = None
        self._add_assist_msg(
            f"⚠️ 出错了：{msg}\n\n"
            "提示：确认 ollama 已启动，或 llama.cpp 模型文件就位，"
            "且本机内存充足。")
        self.input.setEnabled(True)
        self.btn_send.setEnabled(True)

    # ---- 页面切换接口 ----
    def on_leave(self) -> None:
        """切离本页：等待未完成的推理结束，释放共享后端。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        if self._agent is not None:
            self._agent.close()
            self._agent = None
