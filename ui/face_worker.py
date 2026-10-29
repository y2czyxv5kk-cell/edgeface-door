# -*- coding: utf-8 -*-
"""后台识别线程：摄像头读帧 → 推理 → 比对 → 画框 → 开门，不阻塞 UI。

信号：
  frame_ready(np.ndarray, list)  每帧一次：画好框的 BGR 画面 + 精简结果
  door_triggered(str)            实际触发开门时发出，参数为人名
  failed(str)                    致命错误（模型加载失败/摄像头打不开等）
"""
import platform
import time
from typing import List, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from edgeface import create_engine
from edgeface.pipeline import FacePipeline
from edgeface.matcher import FaceMatcher, DoorCooldown
from edgeface.door import DoorController, FakeDoor
from edgeface.draw import draw_face_box
from edgeface.log import RecognitionLog


def _open_camera(src):
    """打开摄像头/视频源。Linux 用 V4L2；Windows 该常量也存在但后端不可用，走默认。"""
    if isinstance(src, str):
        return cv2.VideoCapture(src)
    if platform.system() == "Linux":
        return cv2.VideoCapture(src, cv2.CAP_V4L2)
    return cv2.VideoCapture(src)


class FaceWorker(QThread):
    """人脸识别 + 开门工作线程。

    用法：worker = FaceWorker(db_emb, id_map, ...) → 连接信号 →
    worker.start()；退出时 worker.stop() + worker.wait()。
    """

    frame_ready = pyqtSignal(np.ndarray, list)
    door_triggered = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, db_emb: np.ndarray, id_map: dict,
                 source=0, sim_thresh: float = 0.50, margin: float = 0.10,
                 cooldown: float = 5.0, det_size: int = 640,
                 use_gpio: bool = False, gpio_pin: int = 17,
                 cam_resolution: Tuple[int, int] = (1280, 720),
                 use_mjpg: bool = True, parent=None):
        """Args:
            db_emb: (N, 512) 特征库矩阵
            id_map: 下标 → 姓名 映射
            source: 摄像头设备号（int）或视频文件路径（str）
            sim_thresh / margin: 双阈值判定（见 edgeface/matcher.py）
            cooldown: 同一人两次开门最小间隔（秒）
            det_size: 检测输入尺寸
            use_gpio: True 走真实 GPIO；False 用 FakeDoor
            gpio_pin: GPIO 引脚号
            cam_resolution: USB 摄像头目标分辨率 (W, H)
            use_mjpg: 切换 MJPG 编码提速
        """
        super().__init__(parent)
        self._db_emb = db_emb
        self._id_map = id_map
        self._source = source
        self._sim_thresh = sim_thresh
        self._margin = margin
        self._cooldown = cooldown
        self._det_size = det_size
        self._use_gpio = use_gpio
        self._gpio_pin = gpio_pin
        self._cam_resolution = cam_resolution
        self._use_mjpg = use_mjpg

        self._stop_flag = False
        self._pipe: FacePipeline = None
        self._cap = None
        self._door = None

    def stop(self) -> None:
        """请求退出。线程在下一次循环检查标志后释放资源。"""
        self._stop_flag = True

    def run(self) -> None:  # noqa: N802（QThread 约定命名）
        """线程主循环。"""
        # 1. 初始化流水线（自动选后端）、比对器、冷却器
        try:
            engine = create_engine(input_size=self._det_size)
            self._pipe = FacePipeline(engine)
            self._matcher = FaceMatcher(self._db_emb, self._id_map,
                                        sim_thresh=self._sim_thresh,
                                        margin=self._margin)
            self._cooldown = DoorCooldown(cooldown_seconds=self._cooldown)
        except Exception as e:
            self.failed.emit(f"模型加载失败：{e}")
            return

        # 初始化期间（模型加载较慢）也可能被请求停止，尽早退出
        if self._stop_flag:
            self._safe_release_all()
            return

        # 2. 开门控制器（GPIO 不可用时降级 FakeDoor）
        try:
            self._door = (DoorController(pin=self._gpio_pin) if self._use_gpio
                          else FakeDoor())
        except Exception as e:
            self._door = FakeDoor()
            print(f"[warn] GPIO 不可用（{e}），已降级 FakeDoor")

        if self._stop_flag:
            self._safe_release_all()
            return

        # 3. 打开摄像头
        src = self._source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self._cap = _open_camera(src)
        print(f"[worker] 视频源打开：{src!r} opened={self._cap.isOpened()}",
              flush=True)
        if not self._cap.isOpened():
            self.failed.emit(f"无法打开视频源：{src}")
            self._safe_release_all()
            return

        # MJPG 提速（仅设备号生效）
        if self._use_mjpg and isinstance(src, int):
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if isinstance(src, int):
            w, h = self._cam_resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if self._stop_flag:
            self._safe_release_all()
            return

        # 验证摄像头真能读出帧
        ok_test, _ = self._cap.read()
        print(f"[worker] 首帧读取：{ok_test}", flush=True)
        if not ok_test:
            self.failed.emit("摄像头打开但读不出帧（被占用 / USB 带宽不够）")
            self._safe_release_all()
            return

        # 4. 主循环：读帧 → 推理 → 比对 → 画框 → 开门
        self._log = RecognitionLog()
        last_stranger_log = 0.0
        is_file = isinstance(src, str)  # 视频文件循环播放；摄像头掉线即报错
        try:
            while not self._stop_flag:
                ok, frame = self._cap.read()
                if not ok:
                    if is_file:
                        # 视频播完回到开头循环（demo 持续展示用）
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.failed.emit("视频源结束/掉线")
                    break

                # max_num=5：每帧最多处理 5 张脸，控制耗时
                results = self._pipe.process(frame, max_num=5)
                now = time.time()

                page_results: List[Tuple[str, float, tuple]] = []
                for r in results:
                    m = self._matcher.match(r.embedding)
                    color = (0, 255, 0) if m.recognized else (0, 0, 255)
                    label = (f"{m.top1_name} {m.top1_sim:.2f}" if m.recognized
                             else f"陌生人 {m.top1_sim:.2f}")
                    frame = draw_face_box(frame, r.bbox, label, color_bgr=color)

                    x1, y1, x2, y2 = r.bbox.astype(int)
                    page_results.append(
                        (m.top1_name if m.recognized else "陌生人",
                         m.top1_sim, (int(x1), int(y1), int(x2), int(y2))))

                    # 命中且过冷却 → 写日志 + 开门脉冲（阻塞约 0.8s）
                    if m.recognized and self._cooldown.allow(m.top1_name):
                        self._log.append(m.top1_name, m.top1_sim, recognized=True)
                        self._door.trigger()
                        self.door_triggered.emit(m.top1_name)
                    elif not m.recognized and now - last_stranger_log > 30:
                        # 陌生人日志节流：30 秒最多记一条
                        last_stranger_log = now
                        self._log.append("陌生人", m.top1_sim, recognized=False)

                self.frame_ready.emit(frame, page_results)
        except Exception as e:
            self.failed.emit(f"识别循环异常：{e}")
        finally:
            self._safe_release_all()

    def _safe_release_all(self) -> None:
        """释放摄像头/引擎/GPIO，各步独立 try。"""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._pipe is not None:
            try:
                self._pipe.release()
            except Exception:
                pass
            self._pipe = None
        if self._door is not None:
            try:
                self._door.close()
            except Exception:
                pass
            self._door = None
