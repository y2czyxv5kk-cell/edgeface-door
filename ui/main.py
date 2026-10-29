# -*- coding: utf-8 -*-
"""程序入口：登录窗口 → 主界面（门禁 / 智能助手）。

运行：python ui/main.py（默认账号 admin / 123456）
"""
import os
import sys

# 路径引导：保证 ui/ 与 src/ 都能被 import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows DLL 冲突修复：PyQt5 先加载会破坏 onnxruntime 的 DLL 初始化
# （DLL 初始化例程失败），必须先加载 onnxruntime（详见 docs/开发实录.md）。
# 板端无 onnxruntime，忽略导入失败即可。
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

from PyQt5.QtGui import QColor, QPalette  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from ui.login_window import LoginWindow  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("EdgeFace 智能门禁")

    # 深色主题下保持输入框占位符可读
    pal = app.palette()
    pal.setColor(QPalette.PlaceholderText, QColor("#475569"))
    app.setPalette(pal)

    player = MainWindow()
    login = LoginWindow()

    def on_login_success(username: str):
        login.close()
        player.set_current_user(username)
        player.showFullScreen()

    login.login_success.connect(on_login_success)
    login.showFullScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
