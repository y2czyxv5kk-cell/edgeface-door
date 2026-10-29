"""结果可视化：画人脸框 + 中文标签。

cv2.putText 不支持中文，标签用 PIL truetype 字体渲染。
字体路径覆盖板端（文泉驿/Noto）、WSL2（/mnt/c）、Windows 三种环境。
"""
from functools import lru_cache
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CJK_FONT_CANDIDATES = [
    # Linux / 板端
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # WSL2（挂载的 Windows 字体）
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    # Windows 原生
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]


@lru_cache(maxsize=1)
def _find_cjk_font_path() -> str:
    """返回第一个存在的 CJK 字体路径；找不到返回空串。"""
    for p in _CJK_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    print("[warn] 未找到 CJK 字体，中文标签将无法正常显示")
    return ""


@lru_cache(maxsize=8)
def _get_cjk_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """按字号取字体对象（缓存）。"""
    path = _find_cjk_font_path()
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text_chinese(img_bgr: np.ndarray, text: str, org,
                      color_bgr, font_size: int = 24) -> np.ndarray:
    """在 BGR 图上绘制中英文文本（原地修改并返回）。

    Args:
        img_bgr: 目标图（BGR, uint8）
        text: 文字（可含中文）
        org: 左上角坐标 (x, y)
        color_bgr: 颜色 (B, G, R)
        font_size: 字号
    """
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_cjk_font(font_size)
    # PIL 用 RGB，调用方按 OpenCV 习惯传 BGR，这里转换
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text(org, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_face_box(img_bgr: np.ndarray, bbox, label: str,
                  color_bgr=(0, 255, 0), thickness: int = 3) -> np.ndarray:
    """画人脸框和框上方标签（原地修改并返回）。

    Args:
        img_bgr: 目标图
        bbox: (4,) xyxy 坐标
        label: 标签文字（如 "张三 0.72" / "陌生人 0.41"）
        color_bgr: 框与文字颜色
        thickness: 框线宽
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)

    # 字号按图宽自适应
    font_size = max(18, int(img_bgr.shape[1] / 60))
    label_y = max(0, y1 - font_size - 4)  # 标签画在框上方，贴顶时压进框内
    return draw_text_chinese(img_bgr, label, (x1, label_y),
                             color_bgr, font_size=font_size)
