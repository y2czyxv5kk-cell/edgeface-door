"""RKNN 推理引擎：RK3588 板端用，检测/识别两个 .rknn 模型跑 NPU。

RKNN 模型的预处理（mean/std=127.5/128.0）在转换时已内置，
板端喂入 [0,255] 原始像素即可，不能重复归一化。
输入布局 NHWC float32，与 OpenCV 读图布局一致。
"""
import time
from typing import List

import cv2
import numpy as np

from .base import FaceEngine, Detection

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None  # 非板端环境无此包，实例化时才报错

from ..postprocess import detect as _postprocess_detect


class RknnEngine(FaceEngine):
    """SCRFD 检测 + MobileFaceNet 识别，两个 RKNNLite 会话。"""

    def __init__(self, det_model_path: str, rec_model_path: str,
                 input_size: int = 640, core=None, nms_iou: float = 0.4):
        """Args:
            det_model_path: SCRFD 检测模型 .rknn 路径
            rec_model_path: MobileFaceNet 识别模型 .rknn 路径
            input_size: 检测输入边长，须与转换模型时一致
            core: NPU 核掩码（RK3566 单核可不用；3588 可传三核）
            nms_iou: 后处理 NMS 的 IoU 阈值
        """
        if RKNNLite is None:
            raise ImportError(
                "找不到 rknnlite-api。从 https://github.com/airockchip/rknn-toolkit2 "
                "下载对应板端 Python 版本的 aarch64 whl 安装。"
            )

        self.input_size = input_size
        self.nms_iou = nms_iou

        self._det_rknn = RKNNLite()
        ret = self._det_rknn.load_rknn(det_model_path)
        if ret != 0:
            raise RuntimeError(f"检测模型 load_rknn 失败 ({ret}): {det_model_path}")
        self._init_runtime(self._det_rknn, core, "检测")

        self._rec_rknn = RKNNLite()
        ret = self._rec_rknn.load_rknn(rec_model_path)
        if ret != 0:
            raise RuntimeError(f"识别模型 load_rknn 失败 ({ret}): {rec_model_path}")
        self._init_runtime(self._rec_rknn, core, "识别")

        # 耗时统计，默认关闭
        self.debug_timing = False
        self._t_pre = 0.0
        self._t_npu = 0.0
        self._t_post = 0.0
        self._n_calls = 0

    @staticmethod
    def _init_runtime(rknn, core, tag: str):
        """初始化 NPU 运行时。core_mask/core 参数名按 rknnlite 版本兼容。"""
        try:
            ret = rknn.init_runtime() if core is None else rknn.init_runtime(core_mask=core)
        except TypeError:
            ret = rknn.init_runtime() if core is None else rknn.init_runtime(core=core)
        if ret != 0:
            raise RuntimeError(f"{tag}模型 init_runtime 失败 ({ret})")

    def enable_debug_timing(self):
        """开启耗时统计，用 dump_timing() 查看。"""
        self.debug_timing = True

    def dump_timing(self, reset: bool = True):
        """打印检测各阶段平均耗时（预处理/NPU/后处理）。"""
        if self._n_calls == 0:
            return
        n = self._n_calls
        print(f"  [rknn detect] n={n}  pre={self._t_pre*1000/n:.0f}ms  "
              f"npu={self._t_npu*1000/n:.0f}ms  post={self._t_post*1000/n:.0f}ms")
        if reset:
            self._t_pre = self._t_npu = self._t_post = 0.0
            self._n_calls = 0

    def detect(self, img_bgr: np.ndarray, det_thresh: float = 0.5,
               max_num: int = 0) -> List[Detection]:
        """检测：等比缩放+黑边填充 → NPU 推理 → 解码+NMS → 坐标缩放回原图。

        预处理与 InsightFace 参考实现一致（letterbox 保持纵横比，
        不直接压成正方形），保证 PC 验证过的阈值在板端同样成立。
        """
        h, w = img_bgr.shape[:2]

        # letterbox：等比缩放到 input_size 内，右下黑边填充（左上对齐）
        t0 = time.time()
        im_ratio = h / float(w)
        if im_ratio > 1.0:
            new_h = self.input_size
            new_w = max(1, int(new_h / im_ratio))
        else:
            new_w = self.input_size
            new_h = max(1, int(new_w * im_ratio))
        det_scale = new_h / float(h)
        resized = cv2.resize(img_bgr, (new_w, new_h))
        det_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized
        rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        blob = rgb[np.newaxis, ...].astype(np.float32)  # (1, H, W, 3)
        t1 = time.time()

        # data_type='float32' 让 INT8 模型输出自动反量化；旧版无此参数则跳过
        try:
            outputs = self._det_rknn.inference(inputs=[blob], data_type='float32')
        except TypeError:
            outputs = self._det_rknn.inference(inputs=[blob])
        t2 = time.time()

        if outputs is None:
            raise RuntimeError(
                f"检测推理返回 None，通常是 input_size={self.input_size} "
                f"与转换模型时的尺寸不一致。"
            )

        # 9 个输出张量 → 解码 → NMS（纯 numpy，见 postprocess.py）
        results = _postprocess_detect(
            outputs, self.input_size, det_thresh, self.nms_iou, max_num,
            img_center_xy=(new_w / 2, new_h / 2))
        t3 = time.time()

        # 坐标从模型输入坐标系缩放回原图坐标系（letterbox 左上对齐，
        # 除以 det_scale 即可，无需减 padding 偏移）
        for r in results:
            r["bbox"] /= det_scale
            r["kps"] /= det_scale

        if self.debug_timing:
            self._t_pre += t1 - t0
            self._t_npu += t2 - t1
            self._t_post += t3 - t2
            self._n_calls += 1

        return [
            Detection(bbox=r["bbox"], kps=r["kps"], score=r["score"])
            for r in results
        ]

    def extract(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """提取 512 维特征。输入 112×112 对齐脸，处理方式同 detect 的预处理。"""
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        blob = rgb[np.newaxis, ...].astype(np.float32)  # (1, 112, 112, 3)

        try:
            outputs = self._rec_rknn.inference(inputs=[blob], data_type='float32')
        except TypeError:
            outputs = self._rec_rknn.inference(inputs=[blob])

        emb = np.asarray(outputs[0], dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def release(self) -> None:
        """释放两个 NPU 会话。"""
        for rknn in (self._det_rknn, self._rec_rknn):
            try:
                rknn.release()
            except Exception:
                pass
