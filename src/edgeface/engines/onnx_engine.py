"""ONNX 推理引擎：PC / WSL2 开发环境用，onnxruntime CPU 推理。

模型使用 InsightFace buffalo_s 预训练套装（首次运行自动下载到
~/.insightface/models/buffalo_s/）：
  - detection:   SCRFD-500m 人脸检测
  - recognition: MobileFaceNet 特征提取
与板端 .rknn 模型参数同源，两端算法行为一致。
"""
import warnings
from typing import List

import cv2
import numpy as np

from .base import FaceEngine, Detection


class OnnxEngine(FaceEngine):
    """基于 insightface FaceAnalysis 的引擎。"""

    def __init__(self, det_size: tuple = (640, 640), ctx_id: int = -1,
                 name: str = "buffalo_s"):
        """Args:
            det_size: 检测输入尺寸 (W, H)，越大越能发现小脸，也越慢
            ctx_id:   onnxruntime 设备编号。-1=CPU，>=0=GPU
            name:     insightface 模型包名
        """
        import onnxruntime as _ort
        from insightface.app import FaceAnalysis

        # 只启用实际可用的执行后端，避免 CUDA 告警
        available = _ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in available] or ["CPUExecutionProvider"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._app = FaceAnalysis(name=name, providers=providers)
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)
        self._ctx_id = ctx_id

        # 识别模型需显式 prepare 后才能手动 session.run（见 extract）
        self._rec_model = self._app.models["recognition"]
        self._rec_model.prepare(ctx_id=ctx_id)

    def detect(self, img_bgr: np.ndarray, det_thresh: float = 0.5,
               max_num: int = 0) -> List[Detection]:
        """SCRFD 检测。insightface 内部完成 resize→归一化→推理→解码→NMS。"""
        faces = self._app.get(img_bgr, max_num=max_num or 0)

        # 置信度后过滤：不依赖 insightface 版本是否支持 det_thresh 参数
        faces = [f for f in faces if float(f.det_score) >= det_thresh]
        faces.sort(key=lambda f: float(f.det_score), reverse=True)

        return [
            Detection(
                bbox=np.asarray(f.bbox, dtype=np.float32),
                kps=np.asarray(f.kps, dtype=np.float32),
                score=float(f.det_score),
            )
            for f in faces
        ]

    def extract(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """提取 512 维特征。

        手动组 blob + session.run（不用 insightface 的 get_feat，
        因为 get_feat 内部会再做一次检测对齐，而这里输入已是对齐脸）。
        """
        m = self._rec_model
        size = (112, 112)

        # scale=1/std 完成 (pixel-mean)/std 归一化；swapRB=1 完成 BGR→RGB
        blob = cv2.dnn.blobFromImage(
            aligned_face_bgr,
            1.0 / float(m.input_std),
            size,
            float(m.input_mean),
            swapRB=1,
        )
        net_out = m.session.run(m.output_names, {m.input_name: blob})[0]

        embedding = np.asarray(net_out, dtype=np.float32).ravel()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def release(self) -> None:
        """FaceAnalysis 无显式释放接口，置空引用。"""
        self._app = None
        self._rec_model = None
