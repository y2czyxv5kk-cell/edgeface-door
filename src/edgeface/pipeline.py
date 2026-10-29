"""人脸识别流水线：检测 → 对齐 → 特征提取。

只依赖 FaceEngine 接口，与具体推理后端解耦；
对齐作为独立步骤执行，可单独验证与计时。
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from .engines.base import FaceEngine
from .aligner import FaceAligner


@dataclass
class FaceResult:
    """一张脸的完整识别结果。

    bbox:      (4,) xyxy 原图坐标
    kps:       (5, 2) 关键点原图坐标
    det_score: float 检测置信度
    embedding: (512,) L2 归一化特征，可直接与特征库做点积比对
    """
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    embedding: np.ndarray


class FacePipeline:
    """检测 → 对齐 → 识别 三步流水线。"""

    def __init__(self, engine: FaceEngine, aligner: FaceAligner = None):
        """Args:
            engine: FaceEngine 实现（RKNN 或 ONNX）
            aligner: 对齐器；None 用默认 112×112 配置
        """
        self.engine = engine
        self.aligner = aligner or FaceAligner(output_size=112)

    def process(self, img_bgr: np.ndarray, det_thresh: float = 0.5,
                max_num: int = 0) -> List[FaceResult]:
        """对一帧图像跑完整流水线。

        Args:
            img_bgr: BGR 图像 (H, W, 3)，uint8
            det_thresh: 检测置信度阈值
            max_num: 最多处理几张脸，0 不限（多人场景建议限 5 控制耗时）
        Returns:
            FaceResult 列表（按检测置信度降序）；无脸返回空列表
        """
        # ① 检测：所有人脸的 bbox + 5 关键点
        detections = self.engine.detect(img_bgr, det_thresh=det_thresh,
                                        max_num=max_num)

        results = []
        for det in detections:
            # ② 对齐：仿射变换到 112×112 标准脸
            aligned = self.aligner.align(img_bgr, det.kps)
            # ③ 识别：提取 512 维归一化特征
            embedding = self.engine.extract(aligned)
            results.append(FaceResult(
                bbox=det.bbox, kps=det.kps,
                det_score=det.score, embedding=embedding,
            ))
        return results

    def release(self) -> None:
        """释放引擎资源。"""
        self.engine.release()
