# -*- coding: utf-8 -*-
"""智能门禁页：实时人脸识别 → 命中 → 开门。

点「开启门禁」启动后台线程（模型加载约 1-2s）；
画面绿框=识别成功（名字+相似度），红框=陌生人；
命中且过冷却触发开门，状态栏闪烁提示。切离页面自动停线程。

环境适配：板端 rknn 后端 + 可选真 GPIO；PC onnx 后端 + FakeDoor；
两者都缺时显示依赖安装提示，按钮置灰。
"""
import json
import os
import sys

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.path.join(_PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from edgeface import BACKEND  # noqa: E402

from .. import common  # noqa: E402
from ..common import TEXT_BTN_STYLE  # noqa: E402
from ..face_worker import FaceWorker  # noqa: E402

_DB_DIR = os.path.join(_PROJECT_ROOT, "assets", "face_db")
_EMB_PATH = os.path.join(_DB_DIR, "embeddings.npy")
_ID_MAP_PATH = os.path.join(_DB_DIR, "id_map.json")

# 板端 ISP 主输出常在 video9；PC 默认 0
_BOARD_PREFERRED_CAMERA = "/dev/video9"


class DoorPage(QWidget):
    """智能门禁页。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageRoot")
        self.setStyleSheet(common.PAGE_STYLE)
        self._worker: FaceWorker = None
        self._db_emb = None
        self._id_map = None
        self._build_ui()

        # 可用性检查：后端 + 特征库文件
        if BACKEND is None:
            print("[door] 初始化：BACKEND=None → 缺少推理后端", flush=True)
            self._show_unavailable(
                "当前环境缺少推理后端",
                "PC/WSL 开发环境：pip install insightface onnxruntime\n"
                "RK3566 板端：安装 rknnlite-api（见 requirements-board.txt）",
            )
        elif not (os.path.isfile(_EMB_PATH) and os.path.isfile(_ID_MAP_PATH)):
            print(f"[door] 初始化：特征库文件缺失 emb={_EMB_PATH!r} "
                  f"idmap={_ID_MAP_PATH!r}", flush=True)
            self._show_unavailable(
                "特征库文件缺失",
                f"请先运行注册工具生成特征库：\n"
                f"  python tools/register.py\n（预期文件：{_DB_DIR}）",
            )
        else:
            self._available = True
            print(f"[door] 初始化：可用（backend={BACKEND}）", flush=True)
            self.info.setText(self._ready_text())

    # ---- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addWidget(common.make_page_header("智能门禁", "人脸识别 · 实时比对 · 开门联动"))

        self.preview = QLabel("待启动")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(420)
        self.preview.setStyleSheet(
            "background-color: #0f1726; color: #64748b; font-size: 18px;"
            "border: 1px solid #1e293b; border-radius: 8px;"
        )
        root.addWidget(self.preview, 1)

        bar = QFrame()
        bar.setObjectName("ctrlbar")
        bar.setFixedHeight(72)
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 10, 0, 10)
        h.setSpacing(10)

        self.btn_start = QPushButton("开启门禁")
        self.btn_stop = QPushButton("停止")
        self.btn_video = QPushButton("选择视频")
        self.btn_cam = QPushButton("摄像头")
        self.btn_stop.setEnabled(False)
        self.info = QLabel("待机")
        self.info.setStyleSheet("color: #64748b; font-size: 15px;")

        for b in (self.btn_start, self.btn_stop, self.btn_video, self.btn_cam):
            b.setStyleSheet(TEXT_BTN_STYLE)
            b.setCursor(Qt.PointingHandCursor)

        h.addWidget(self.btn_start)
        h.addWidget(self.btn_stop)
        h.addWidget(self.btn_video)
        h.addWidget(self.btn_cam)
        h.addStretch(1)
        h.addWidget(self.info)
        root.addWidget(bar)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_video.clicked.connect(self._pick_video)
        self.btn_cam.clicked.connect(self._use_camera)

    def _show_unavailable(self, title: str, detail: str) -> None:
        """不可用时提示打到预览区，按钮置灰。"""
        self._available = False
        self.preview.setText(f"{title}\n\n{detail}")
        self.btn_start.setEnabled(False)
        self.info.setText(title)

    # ---- 启动 / 停止 ----
    def _ensure_db_loaded(self) -> bool:
        """惰性加载特征库（首次启动时读盘，之后用缓存）。"""
        if self._db_emb is not None:
            return True
        try:
            self._db_emb = np.load(_EMB_PATH)
            with open(_ID_MAP_PATH, encoding="utf-8") as f:
                self._id_map = {int(k): v for k, v in json.load(f).items()}
        except Exception as e:
            print(f"[door] 特征库加载异常：{e!r}", flush=True)
            QMessageBox.warning(self, "错误", f"加载特征库失败：{e}")
            return False
        return True

    def _pick_video(self):
        """选择演示视频文件作为识别源（视频文件会循环播放）。"""
        start_dir = os.path.join(_PROJECT_ROOT, "testvideo")
        if not os.path.isdir(start_dir):
            start_dir = ""
        print("[door] 打开文件选择框…", flush=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "选择演示视频", start_dir,
            "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)")
        print(f"[door] 选择结果：{path!r}", flush=True)
        if not path:
            return
        self._video_path = path
        print(f"[door] 视频源已设为 {path}，自动开始识别", flush=True)
        self._start()

    def _use_camera(self):
        """切回摄像头源。"""
        self._video_path = None
        self.info.setText("摄像头源")

    def _resolve_camera_source(self):
        """视频文件优先；板端 /dev/video9；其它环境默认 0。"""
        if getattr(self, "_video_path", None):
            return self._video_path
        if os.path.exists(_BOARD_PREFERRED_CAMERA):
            return _BOARD_PREFERRED_CAMERA
        return 0

    def _start(self):
        print(f"[door] _start 进入，available="
              f"{getattr(self, '_available', None)!r}", flush=True)
        if not getattr(self, "_available", False):
            print("[door] _start 早退：available=False", flush=True)
            return
        if not self._ensure_db_loaded():
            print("[door] _start 早退：特征库加载失败", flush=True)
            return
        if self._worker is not None and self._worker.isRunning():
            print("[door] _start 早退：worker 已在运行", flush=True)
            return
        print(f"[door] 启动识别，源={self._resolve_camera_source()!r}", flush=True)

        # 板端装 gpiod 才走真 GPIO；PC 演示一律 FakeDoor
        use_gpio = False
        if BACKEND == "rknn":
            try:
                import gpiod  # noqa: F401
                use_gpio = True
            except ImportError:
                pass

        # 安全：视频文件是回放不是真人，一律禁止驱动真实门锁
        # （板端也是：只有实时摄像头源才能触发真 GPIO）
        source = self._resolve_camera_source()
        if isinstance(source, str):
            use_gpio = False

        self._worker = FaceWorker(
            db_emb=self._db_emb,
            id_map=self._id_map,
            source=source,
            use_gpio=use_gpio,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.door_triggered.connect(self._on_door_triggered)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.info.setText("识别中…")
        self.info.setStyleSheet("color: #22d3ee; font-size: 15px;")

    def _ready_text(self) -> str:
        """初始状态栏文案。"""
        return (f"就绪 · 后端 {BACKEND.upper()}"
                f"（{'NPU' if BACKEND == 'rknn' else 'CPU'}）")

    def _reset_to_initial(self) -> None:
        """恢复到页面初始状态：清视频源/特征缓存，预览回占位，按钮复位。"""
        # 清空视频源选择：停止后需重新选视频才能回放，
        # 防止「停止→开启」直接回放旧视频触发开门（安全）
        self._video_path = None
        # 清特征库缓存：下次启动重新读盘（注册新人后无需重启应用）
        self._db_emb = None
        self._id_map = None
        self.btn_start.setEnabled(getattr(self, "_available", False))
        self.btn_stop.setEnabled(False)
        self.info.setStyleSheet("color: #64748b; font-size: 15px;")
        if getattr(self, "_available", False):
            self.preview.setPixmap(QPixmap())
            self.preview.setText("待启动")
            self.info.setText(self._ready_text())
        # 不可用状态保留 _show_unavailable 的提示文案，不覆盖

    def stop(self) -> None:
        """停 worker 并等线程退出，页面恢复初始状态。"""
        if self._worker is None:
            return
        print("[door] stop：等待 worker 退出…", flush=True)
        self._worker.stop()
        self._worker.wait(5000)
        self._worker = None
        self._reset_to_initial()
        print("[door] stop：已恢复初始状态", flush=True)

    # ---- worker 信号 ----
    def _on_frame(self, frame_bgr: np.ndarray, results: list) -> None:
        """每帧画面：BGR ndarray → QPixmap 上屏。

        停止后 worker 已置 None，此时到达的迟到帧直接丢弃，
        防止把已复位的「待启动」预览又画回旧画面。
        """
        if self._worker is None:
            return
        rgb = frame_bgr[..., ::-1].copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix.scaled(self.preview.size(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation))

    def _on_door_triggered(self, name: str) -> None:
        """开门成功：状态栏闪绿色提示 2s。"""
        self.info.setText(f"✅ 已开门：{name}")
        self.info.setStyleSheet("color: #22c55e; font-size: 15px; font-weight: 600;")
        QTimer.singleShot(2000, lambda: (
            self.info.setText("识别中…"),
            self.info.setStyleSheet("color: #22d3ee; font-size: 15px;"),
        ))

    def _on_failed(self, msg: str) -> None:
        """worker 报错：自动停止 + 弹窗。"""
        self.stop()
        QMessageBox.warning(self, "门禁错误", msg)

    # ---- 页面切换接口 ----
    def on_leave(self) -> None:
        """切离本页：停 worker 释放摄像头/NPU。"""
        self.stop()
