"""图片人脸识别：单图/目录 → 检测对齐识别 → 特征库比对 → 画框标注。

判定规则与实时门禁一致：top1 相似度 >= 0.50 且 top1-top2 >= 0.10
才算识别成功，否则标"陌生人"（红框）。结果图存 runs/recognize/。

用法：
    python tools/recognize.py test.jpg
    python tools/recognize.py photos/
    python tools/recognize.py test.jpg --sim-thresh 0.6
"""
import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import cv2  # noqa: E402

from edgeface import create_engine  # noqa: E402
from edgeface.pipeline import FacePipeline  # noqa: E402
from edgeface.matcher import FaceMatcher  # noqa: E402
from edgeface.draw import draw_face_box  # noqa: E402
from edgeface.io import imread, imwrite  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_DEFAULT_DB = os.path.join(_PROJECT_ROOT, "assets", "face_db")
_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def _list_images(path: str):
    """目录 → 图片列表；单文件 → 单元素列表。"""
    if os.path.isdir(path):
        files = []
        for ext in _IMG_EXTS:
            files.extend(glob.glob(os.path.join(path, ext)))
        return sorted(files)
    return [path] if os.path.isfile(path) else []


def main():
    parser = argparse.ArgumentParser(description="图片人脸识别")
    parser.add_argument("input", nargs="?", default=os.path.join(_PROJECT_ROOT, "test_pic"),
                        help="识别目标：单图或目录")
    parser.add_argument("--db", default=_DEFAULT_DB, help="特征库目录")
    parser.add_argument("--sim-thresh", type=float, default=0.50, help="相似度阈值")
    parser.add_argument("--margin", type=float, default=0.10, help="top1-top2 差距阈值")
    parser.add_argument("--det-size", type=int, default=640, help="检测输入尺寸")
    parser.add_argument("--save", action="store_true", default=True,
                        help="结果图存到 runs/recognize/（默认开）")
    parser.add_argument("--no-save", dest="save", action="store_false")
    args = parser.parse_args()

    try:
        matcher = FaceMatcher.from_db_dir(
            args.db, sim_thresh=args.sim_thresh, margin=args.margin)
    except FileNotFoundError as e:
        print(e)
        print("请先运行注册工具：python tools/register.py")
        sys.exit(1)

    print(f"加载特征库：{len(matcher.db)} 人")
    print(f"判定规则：top1_sim >= {args.sim_thresh} 且 top1-top2 >= {args.margin}")

    print("初始化流水线…")
    engine = create_engine(input_size=args.det_size)
    pipe = FacePipeline(engine)

    images = _list_images(args.input)
    if not images:
        print(f"未找到图片：{args.input}")
        sys.exit(1)
    print(f"共 {len(images)} 张图片待识别\n")

    save_dir = None
    if args.save:
        save_dir = Path(_PROJECT_ROOT) / "runs" / "recognize"
        save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        img = imread(img_path)  # 中文路径安全
        if img is None:
            print(f"[{os.path.basename(img_path)}] 读取失败")
            continue
        results = pipe.process(img)

        if not results:
            print(f"[{os.path.basename(img_path)}] 未检测到人脸")
            continue

        print(f"[{os.path.basename(img_path)}] {len(results)} 个人脸:")
        for i, r in enumerate(results):
            m = matcher.match(r.embedding)
            decision = f"→ {m.top1_name}" if m.recognized else "→ 陌生人"

            x1, y1, x2, y2 = r.bbox.astype(int)
            print(f"  #{i} top1={m.top1_name}({m.top1_sim:.3f}) "
                  f"top2={m.top2_sim:.3f} det={r.det_score:.2f} "
                  f"[{x1},{y1},{x2},{y2}] {decision}")

            if save_dir is not None:
                color = (0, 255, 0) if m.recognized else (0, 0, 255)
                label = (f"{m.top1_name} {m.top1_sim:.2f}" if m.recognized
                         else f"陌生人 {m.top1_sim:.2f}")
                img = draw_face_box(img, r.bbox, label, color_bgr=color)

        if save_dir is not None:
            out_path = save_dir / f"result_{os.path.basename(img_path)}"
            imwrite(str(out_path), img)  # 中文路径安全
            print(f"  → 可视化已保存：{out_path}")

    pipe.release()


if __name__ == "__main__":
    main()
