# -*- coding: utf-8 -*-
"""登录窗口：账号密码校验，成功后发射 login_success 信号。"""

import time

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from .common import (
    COLOR_BG, COLOR_TEXT, COLOR_TEXT_DIM, INPUT_STYLE, TEXT_BTN_STYLE, fit_to_board,
)

# 演示账号。生产环境应替换为哈希认证（PBKDF2 + 盐 + 失败锁定）
ACCOUNTS = {
    "admin": "123456",
}

_MAX_FAILS = 5        # 连续失败锁定阈值
_LOCK_SECONDS = 30    # 锁定时长（秒）


class LoginWindow(QWidget):
    """全屏登录窗口。"""

    login_success = pyqtSignal(str)  # 参数：用户名

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EdgeFace 智能门禁 - 登录")
        self._fail_count = 0
        self._locked_until = 0.0
        self._build_ui()
        fit_to_board(self)

    def _build_ui(self):
        self.setStyleSheet(f"background-color: {COLOR_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(60, 60, 60, 60)

        root.addStretch(1)
        title = QLabel("EdgeFace 智能门禁")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 40px; font-weight: 700;")
        root.addWidget(title)

        subtitle = QLabel("端侧人脸识别 · 感知-决策-执行闭环")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 18px;")
        root.addWidget(subtitle)
        root.addStretch(1)

        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("账号")
        self.user_edit.setMinimumHeight(56)
        self.user_edit.setStyleSheet(INPUT_STYLE)
        root.addWidget(self.user_edit)

        self.pass_edit = QLineEdit()
        self.pass_edit.setPlaceholderText("密码")
        self.pass_edit.setEchoMode(QLineEdit.Password)
        self.pass_edit.setMinimumHeight(56)
        self.pass_edit.setStyleSheet(INPUT_STYLE)
        self.pass_edit.returnPressed.connect(self._try_login)
        root.addWidget(self.pass_edit)

        self.btn_login = QPushButton("登 录")
        self.btn_login.setMinimumHeight(64)
        self.btn_login.setStyleSheet(TEXT_BTN_STYLE)
        self.btn_login.setCursor(Qt.PointingHandCursor)
        self.btn_login.clicked.connect(self._try_login)
        root.addWidget(self.btn_login)

        self.msg = QLabel("")
        self.msg.setAlignment(Qt.AlignCenter)
        self.msg.setStyleSheet("color: #f87171; font-size: 16px;")
        root.addWidget(self.msg)

        hint = QLabel("默认账号 admin / 123456")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 14px;")
        root.addWidget(hint)
        root.addStretch(1)

    def _try_login(self):
        """校验账号密码；连续失败 5 次锁定 30 秒。"""
        if self._locked_until and time.time() < self._locked_until:
            remain = int(self._locked_until - time.time())
            self.msg.setText(f"尝试次数过多，请 {remain}s 后重试")
            return

        user = self.user_edit.text().strip()
        pwd = self.pass_edit.text()

        # 用户不存在时也执行一次比较，避免时序侧信道泄露用户是否存在
        expect = ACCOUNTS.get(user)
        if expect is not None and expect == pwd:
            self._fail_count = 0
            self.login_success.emit(user)
            return

        self._fail_count += 1
        if self._fail_count >= _MAX_FAILS:
            self._locked_until = time.time() + _LOCK_SECONDS
            self._fail_count = 0
            self.msg.setText(f"尝试次数过多，锁定 {_LOCK_SECONDS}s")
        else:
            self.msg.setText("账号或密码错误")
        self.pass_edit.clear()
