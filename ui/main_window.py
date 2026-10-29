# -*- coding: utf-8 -*-
"""主窗口：左侧菜单 + 右侧分页（门禁 / 智能助手）。

切离某页时调用其 on_leave()，保证同一时刻只有一个页面占用
摄像头/NPU/LLM 等独占资源。
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from .common import (  # noqa: E402
    COLOR_PANEL, COLOR_SIDEBAR, COLOR_TEXT_DIM, MENU_BTN_STYLE, fit_to_board,
)
from .pages.door_page import DoorPage  # noqa: E402
from .pages.assistant_page import AssistantPage  # noqa: E402


class MainWindow(QWidget):
    """登录后的主界面。"""

    def __init__(self, username: str = ""):
        super().__init__()
        self.setWindowTitle("EdgeFace 智能门禁")
        self._username = username
        self._build_ui()
        fit_to_board(self)

    def _build_ui(self):
        self.setStyleSheet(f"background-color: {COLOR_PANEL};")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 左侧菜单栏
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet(f"background-color: {COLOR_SIDEBAR};")
        side_lay = QVBoxLayout(sidebar)
        side_lay.setContentsMargins(0, 24, 0, 16)
        side_lay.setSpacing(4)

        brand = QLabel("EdgeFace")
        brand.setAlignment(Qt.AlignCenter)
        brand.setStyleSheet("color: #22d3ee; font-size: 24px; font-weight: 700;")
        side_lay.addWidget(brand)
        side_lay.addSpacing(12)

        self.btn_door = QPushButton("门禁")
        self.btn_assist = QPushButton("智能助手")
        for b in (self.btn_door, self.btn_assist):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(MENU_BTN_STYLE)
            side_lay.addWidget(b)
        self.btn_door.setChecked(True)

        side_lay.addStretch(1)
        self._user_label = QLabel(self._username or "未登录")
        self._user_label.setAlignment(Qt.AlignCenter)
        self._user_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 14px;")
        side_lay.addWidget(self._user_label)
        root.addWidget(sidebar)

        # 右侧分页容器
        self.stack = QStackedWidget()
        self.door_page = DoorPage(self)
        self.assistant_page = AssistantPage(self)
        self.stack.addWidget(self.door_page)      # index 0
        self.stack.addWidget(self.assistant_page)  # index 1
        root.addWidget(self.stack, 1)

        self.btn_door.clicked.connect(lambda: self._switch_page(0))
        self.btn_assist.clicked.connect(lambda: self._switch_page(1))

    def set_current_user(self, username: str) -> None:
        """登录成功后更新侧栏用户名。"""
        self._username = username
        self._user_label.setText(username)

    def _switch_page(self, index: int):
        """切页：旧页 on_leave() 释放资源，再切换并同步菜单选中态。"""
        old = self.stack.currentWidget()
        if old is not None and hasattr(old, "on_leave"):
            old.on_leave()
        self.stack.setCurrentIndex(index)
        self.btn_door.setChecked(index == 0)
        self.btn_assist.setChecked(index == 1)

    def closeEvent(self, event):  # noqa: N802（Qt 约定命名）
        """窗口关闭：停后台线程、释放两页资源。"""
        self.door_page.on_leave()
        self.assistant_page.on_leave()
        super().closeEvent(event)
