"""人脸注册：照片目录 → 提取特征 → 生成特征库。

输入目录约定（face_db_raw/<人名>/*.jpg），每人 5~10 张，一张照片
只能有本人一张正脸。每人多张特征取平均再归一化后入库。
face_db_raw 默认放在**项目文件夹外**（个人照片不入 Git 仓库）。

用法：
    python tools/register.py
    python tools/register.py --raw D:/photos --out assets/face_db
"""
import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import numpy as np  # noqa: E402

from edgeface import create_engine  # noqa: E402
from edgeface.pipeline import FacePipeline  # noqa: E402
from edgeface.io import imread  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_DEFAULT_RAW = os.path.join(os.path.dirname(_PROJECT_ROOT), "face_db_raw")
_DEFAULT_OUT = os.path.join(_PROJECT_ROOT, "assets", "face_db")

_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def _list_images(folder: str):
    """列出目录下所有图片（按扩展名过滤，排序保证稳定）。"""
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="人脸库注册")
    parser.add_argument("--raw", default=_DEFAULT_RAW, help="注册输入目录")
    parser.add_argument("--out", default=_DEFAULT_OUT, help="特征库输出目录")
    parser.add_argument("--det-size", type=int, default=640, help="检测输入尺寸")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    if not raw_dir.exists():
        print(f"注册目录不存在：{raw_dir}")
        print("期望布局：face_db_raw/<人名>/<照片>.jpg")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("初始化流水线（首次运行自动下载 buffalo_s 模型，约 250MB）…")
    engine = create_engine(input_size=args.det_size)
    pipe = FacePipeline(engine)

    id_map = {}            # 下标 → 姓名
    embeddings_list = []   # 每人一个 (512,) 平均特征

    # 一个子目录 = 一个人
    persons = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]
    if not persons:
        print(f"在 {raw_dir} 下未发现任何人员子目录")
        sys.exit(1)

    for pid, person_dir in enumerate(persons):
        name = person_dir.name
        imgs = _list_images(str(person_dir))
        if not imgs:
            print(f"[{name}] 跳过：未找到图片")
            continue

        embs, used = [], 0
        for img_path in imgs:
            img = imread(img_path)  # 中文路径安全
            if img is None:
                print(f"  [{name}/{os.path.basename(img_path)}] 读取失败")
                continue
            results = pipe.process(img)
            if not results:
                print(f"  [{name}/{os.path.basename(img_path)}] 未检测到脸")
                continue
            # 多人合照取面积最大的脸（假设是本人）
            best = max(results, key=lambda r: (r.bbox[2] - r.bbox[0])
                       * (r.bbox[3] - r.bbox[1]))
            embs.append(best.embedding)
            used += 1

        if not embs:
            print(f"[{name}] 无可用特征，未入库")
            continue

        # 多张特征平均 → L2 归一化（库内向量模长 1，点积即余弦相似度）
        avg = np.mean(embs, axis=0)
        avg = avg / np.linalg.norm(avg)
        id_map[pid] = name
        embeddings_list.append(avg)
        print(f"[{name}] 入库：{used}/{len(imgs)} 张有效")

    if not embeddings_list:
        print("没有任何人入库，终止")
        sys.exit(1)

    embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)  # (N, 512)
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print(f"\n注册完成：{len(id_map)} 人 → {out_dir}")
    print(f"  embeddings.npy 形状: {embeddings.shape}")
    print(f"  id_map.json: {id_map}")
    pipe.release()


if __name__ == "__main__":
    main()
