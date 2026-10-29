"""引擎抽象层：定义统一推理接口与检测结果结构。

onnx_engine.py（PC 开发）与 rknn_engine.py（板端部署）实现同一接口，
上层流水线只依赖本文件，不依赖具体推理库。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Detection:
    """单张人脸检测结果。

    bbox:  (4,) float32，xyxy 格式，原图像素坐标
    kps:   (5, 2) float32，5 个关键点，原图坐标。
           顺序：左眼中心/右眼中心/鼻尖/左嘴角/右嘴角
    score: float，检测置信度，范围 0~1
    """
    bbox: np.ndarray
    kps: np.ndarray
    score: float


class FaceEngine(ABC):
    """人脸推理引擎接口。

    detect() 输入 BGR 图，返回 Detection 列表（坐标已换算回原图坐标系）；
    extract() 输入 112×112 对齐人脸，返回 (512,) L2 归一化特征。
    """

    @abstractmethod
    def detect(self, img_bgr: np.ndarray, det_thresh: float = 0.5,
               max_num: int = 0) -> List[Detection]:
        """检测图像中所有人脸。

        Args:
            img_bgr: BGR 图像 (H, W, 3)，uint8，值域 0~255
            det_thresh: 置信度下限，低于此值的结果丢弃
            max_num: 最多保留的检测数（按置信度降序），0 表示不限
        Returns:
            Detection 列表；未检测到人脸返回空列表
        """

    @abstractmethod
    def extract(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """对 112×112 对齐人脸提取 512 维特征。

        Args:
            aligned_face_bgr: 112×112 BGR 图像（aligner 的输出）
        Returns:
            (512,) float32，已 L2 归一化（模长 1，点积即余弦相似度）
        """

    def release(self) -> None:
        """释放底层资源，子类按需覆盖。"""
