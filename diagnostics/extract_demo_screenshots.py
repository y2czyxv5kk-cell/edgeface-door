# -*- coding: utf-8 -*-
"""从标注视频抽取 demo 截图：assets/screenshots/demo_frame_*.png。

用法：python diagnostics/extract_demo_screenshots.py [帧号...]
默认抽 30/120/300/500 帧。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2  # noqa: E402

from edgeface import io as edge_io  # noqa: E402

_VIDEO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "testvideo", "ac678b2e5b13a18c54c97ee9f0159e9e_annotated.mp4")
_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "assets", "screenshots")


def main():
    frames = [int(x) for x in sys.argv[1:]] or [30, 120, 300, 500]
    os.makedirs(_OUT, exist_ok=True)
    cap = cv2.VideoCapture(_VIDEO)
    for fno in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            print(f"帧 {fno}: 读取失败")
            continue
        path = os.path.join(_OUT, f"demo_frame_{fno}.png")
        edge_io.imwrite(path, frame)
        print(f"已保存 {path} ({frame.shape[1]}x{frame.shape[0]})")
    cap.release()


if __name__ == "__main__":
    main()
