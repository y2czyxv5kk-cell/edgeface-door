"""人脸对齐：5 关键点 → 仿射变换 → 112×112 标准脸。

用最小二乘求解仿射矩阵，把检测到的 5 个关键点变换到 ArcFace 标准
模板坐标上，再做 warpAffine。纯 numpy/cv2 实现，PC 与板端行为一致。
"""
import numpy as np
import cv2

# ArcFace 标准 5 点模板（112×112 图上的目标坐标）。
# 与 InsightFace 训练 MobileFaceNet 时一致，不可修改。
_ARCFACE_REF_112 = np.array([
    [38.2946, 51.6963],   # 左眼中心
    [73.5318, 51.5014],   # 右眼中心
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 左嘴角
    [70.7299, 92.2041],   # 右嘴角
], dtype=np.float32)


class FaceAligner:
    """5 点人脸对齐器。

    aligner = FaceAligner(output_size=112)
    aligned = aligner.align(img_bgr, kps)   # kps 来自 Detection.kps
    """

    def __init__(self, output_size: int = 112):
        """Args:
            output_size: 对齐后边长。MobileFaceNet 训练用 112，一般不换。
        """
        self.output_size = output_size
        if output_size == 112:
            self._ref = _ARCFACE_REF_112
        else:
            self._ref = _ARCFACE_REF_112 * (output_size / 112.0)

    def align(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """把关键点对齐到标准模板。

        Args:
            img: 原图 BGR (H, W, 3)
            kps: (5, 2) 关键点，原图坐标
        Returns:
            output_size×output_size BGR 图（uint8）

        实现：解 src_homo @ M.T = ref 的最小二乘（5 点 6 未知数，超定），
        得仿射矩阵 M (2,3)，再 warpAffine。
        """
        src = np.asarray(kps, dtype=np.float32).reshape(5, 2)
        # 齐次坐标：每点补 1，构成 (5, 3)
        src_homo = np.hstack([src, np.ones((5, 1), dtype=np.float32)])
        M = np.linalg.lstsq(src_homo, self._ref, rcond=None)[0].T.astype(np.float32)
        return cv2.warpAffine(img, M, (self.output_size, self.output_size),
                              borderValue=0.0)
