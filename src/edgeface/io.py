"""图片读写：兼容中文路径（cv2.imread/imwrite 在 Windows 上不支持中文路径）。

实现：用 numpy 读字节流 + cv2.imdecode 解码；写入用 imencode + tofile。
全项目图片读写统一走本模块。
"""
import os

import cv2
import numpy as np


def imread(path: str):
    """读取图片，兼容中文/空格路径。

    Returns:
        BGR ndarray；失败返回 None（与 cv2.imread 行为一致）
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: str, img: np.ndarray) -> bool:
    """写入图片，兼容中文/空格路径。目标目录须已存在。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True
