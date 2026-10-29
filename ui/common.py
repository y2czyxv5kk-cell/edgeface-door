# -*- coding: utf-8 -*-
"""共享样式与工具：配色、按钮/输入框样式、窗口适配。"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget

# 目标屏幕：板端 800×1080 竖屏；PC 上等比缩放居中
BOARD_W, BOARD_H = 800, 1080

COLOR_BG = "#0a0f1c"        # 主背景
COLOR_PANEL = "#0f1726"     # 面板/卡片
COLOR_SIDEBAR = "#0b1120"   # 侧栏
COLOR_ACCENT = "#22d3ee"    # 青色强调
COLOR_TEXT = "#e2e8f0"      # 主文本
COLOR_TEXT_DIM = "#64748b"  # 次要文本

# 侧栏菜单按钮（checkable，选中态带青色左侧光条）
MENU_BTN_STYLE = """
QPushButton {
    text-align: left;
    padding: 18px 22px;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0px;
    color: #94a3b8;
    font-size: 22px;
    background: transparent;
}
QPushButton:hover    { color: #e2e8f0; background: #111c30; }
QPushButton:checked  {
    color: #ffffff;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(34,211,238,0.22), stop:1 rgba(34,211,238,0.02));
    border-left: 3px solid #22d3ee;
    font-weight: 600;
}
"""

# 主操作按钮
TEXT_BTN_STYLE = """
QPushButton {
    color: #ffffff;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0891b2, stop:1 #2563eb);
    border: 1px solid #22d3ee;
    border-radius: 5px;
    padding: 10px 22px;
    font-size: 16px;
    font-weight: 600;
}
QPushButton:hover    { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #06b6d4, stop:1 #3b82f6); }
QPushButton:pressed  { background: #0e7490; }
QPushButton:disabled { color: #5b6b85; background: #1e293b; border: 1px solid #243049; }
"""

# 输入框
INPUT_STYLE = """
QLineEdit {
    background: #0f1726;
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 14px 16px;
    font-size: 18px;
    selection-background-color: #22d3ee;
    selection-color: #0a0f1c;
}
QLineEdit:focus { border: 1px solid #22d3ee; }
"""

# 内容页根容器
PAGE_STYLE = "QWidget#pageRoot { background-color: #0a0f1c; }"


def fit_to_board(window):
    """按目标分辨率适配窗口大小并居中（板端占满，PC 等比缩小）。"""
    screen = QApplication.primaryScreen()
    if screen is None:
        window.resize(BOARD_W, BOARD_H)
        return
    avail = screen.availableGeometry()
    w = min(BOARD_W, avail.width())
    h = min(BOARD_H, avail.height())
    window.resize(w, h)
    window.move(avail.left() + (avail.width() - w) // 2,
                avail.top() + (avail.height() - h) // 2)


def make_page_header(title: str, subtitle: str = None) -> QWidget:
    """页面标题块：青色标题 + 渐变光带分隔线。"""
    box = QWidget()
    box.setStyleSheet("background: transparent;")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)

    t = QLabel(title)
    t.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 28px; font-weight: 700; background: transparent;")
    lay.addWidget(t)

    if subtitle:
        s = QLabel(subtitle)
        s.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 16px; background: transparent;")
        lay.addWidget(s)

    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(
        "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        "stop:0 #22d3ee, stop:0.6 rgba(34,211,238,0.25), stop:1 rgba(34,211,238,0));"
        "border: none;"
    )
    lay.addWidget(line)
    return box
