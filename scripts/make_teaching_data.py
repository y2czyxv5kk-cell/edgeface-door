# -*- coding: utf-8 -*-
"""生成教学数据包 teaching_data/：标注练习素材 + 模型预标注 + 训练数据集。

数据来源：face_db_raw 注册照 + testvideo 测试视频抽帧。
标注由本项目 SCRFD 检测模型**预标注**生成（工业界"模型预标注+人工校正"
流程的教学素材），学生在标注练习中学会检查与校正。

teaching_data 与 face_db_raw 均位于**项目文件夹外**（教学数据/个人照片
不入 Git 仓库）。

用法：python scripts/make_teaching_data.py
"""
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from edgeface import create_engine  # noqa: E402
from edgeface.pipeline import FacePipeline  # noqa: E402
from edgeface import io as edge_io  # noqa: E402

_TEACHING = _PROJECT_ROOT.parent / "teaching_data"
_TODO_DIR = _TEACHING / "annotate_todo"        # 待标注练习（无标签）
_REF_DIR = _TEACHING / "annotate_ref"          # 参考标注示范（5 张带标签）
_DS_DIR = _TEACHING / "yolo_dataset"           # 训练用完整数据集（预标注）
_LLM_DIR = _TEACHING / "llm_finetune"          # LLM 微调数据样例
_N_REF = 5
_VAL_RATIO = 0.25


def _gather_source_images():
    """收集源图片：注册照 + 视频抽帧，返回 [(路径, BGR图)]。"""
    images = []
    raw_dir = _PROJECT_ROOT.parent / "face_db_raw"
    for f in sorted(raw_dir.rglob("*")):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        img = edge_io.imread(str(f))
        if img is not None:
            images.append((f, img))
    # 视频抽帧：等间隔 12 帧，覆盖整段视频
    vids = sorted((_PROJECT_ROOT / "testvideo").glob("*.mp4"))
    if vids:
        cap = cv2.VideoCapture(str(vids[0]))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n_total:
            step = max(1, n_total // 13)
            for fno in range(step, n_total - 1, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                ok, frame = cap.read()
                if ok:
                    images.append((vids[0], frame))
        cap.release()
    return images


def _detect_faces(pipe, img):
    """检测人脸，返回 YOLO 格式标注行列表（class=0, cx cy w h 归一化）。"""
    h, w = img.shape[:2]
    lines = []
    for r in pipe.process(img, det_thresh=0.5, max_num=10):
        x1, y1, x2, y2 = r.bbox.astype(float)
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _write_yolo_label(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    if _TEACHING.exists():
        print(f"目标目录已存在：{_TEACHING}，先删除旧数据包")
        shutil.rmtree(_TEACHING)
    for d in (_TODO_DIR, _REF_DIR, _DS_DIR / "images" / "train",
              _DS_DIR / "images" / "val", _DS_DIR / "labels" / "train",
              _DS_DIR / "labels" / "val", _LLM_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("加载检测引擎（预标注用）…")
    engine = create_engine(input_size=640)
    pipe = FacePipeline(engine)

    images = _gather_source_images()
    print(f"源图片 {len(images)} 张，开始预标注…")

    records = []  # (序号, 来源, 图片, 标注行)
    for i, (src, img) in enumerate(images):
        lines = _detect_faces(pipe, img)
        if not lines:
            print(f"  [跳过] 第 {i+1} 张未检测到人脸")
            continue
        records.append((i + 1, src, img, lines))

    # 1) 待标注练习：所有图，不给标签
    for idx, src, img, _ in records:
        edge_io.imwrite(str(_TODO_DIR / f"{idx:02d}.jpg"), img)

    # 2) 参考标注示范：前 5 张带预标注标签
    for idx, src, img, lines in records[:_N_REF]:
        edge_io.imwrite(str(_REF_DIR / f"{idx:02d}.jpg"), img)
        _write_yolo_label(_REF_DIR / f"{idx:02d}.txt", lines)

    # 3) 训练数据集：全部预标注，按比例分 train/val（保持顺序切分，简单可复现）
    n_val = max(1, int(len(records) * _VAL_RATIO))
    for i, (idx, src, img, lines) in enumerate(records):
        split = "val" if i < n_val else "train"
        edge_io.imwrite(str(_DS_DIR / "images" / split / f"{idx:02d}.jpg"), img)
        _write_yolo_label(_DS_DIR / "labels" / split / f"{idx:02d}.txt", lines)

    (_DS_DIR / "face.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: face\n",
        encoding="utf-8")

    # 4) LLM 微调数据样例（Alpaca 格式，与 edgebrain 工具协议对齐）
    llm_samples = [
        {"instruction": "帮我开门", "input": "", "output": '{"name": "open_door", "arguments": {"reason": "用户请求"}}'},
        {"instruction": "门开了吗？查一下最近的识别记录", "input": "", "output": '{"name": "recent_logs", "arguments": {"limit": 5}}'},
        {"instruction": "有哪些人注册了人脸？", "input": "", "output": '{"name": "list_users", "arguments": {}}'},
        {"instruction": "请查一下刘昊在不在名单里", "input": "", "output": '{"name": "list_users", "arguments": {"keyword": "刘昊"}}'},
        {"instruction": "你好，你是谁？", "input": "", "output": "你好，我是智能门禁助手，可以帮你开门、查询识别记录和注册名单。"},
        {"instruction": "请帮我开门", "input": "", "output": '{"name": "open_door", "arguments": {}}'},
        {"instruction": "把最近10条门禁记录给我看看", "input": "", "output": '{"name": "recent_logs", "arguments": {"limit": 10}}'},
        {"instruction": "系统里一共注册了几个人？", "input": "", "output": '{"name": "list_users", "arguments": {}}'},
        {"instruction": "我要出去，开门", "input": "", "output": '{"name": "open_door", "arguments": {"reason": "用户要出门"}}'},
        {"instruction": "刚才谁开门进来了？", "input": "", "output": '{"name": "recent_logs", "arguments": {"limit": 3}}'},
        {"instruction": "介绍一下这个系统", "input": "", "output": "这是部署在 RK3588 边缘设备上的智能门禁系统：摄像头实时检测人脸，识别注册人员后自动开门，同时支持用自然语言查询记录和名单。"},
        {"instruction": "张伟在名单里吗？", "input": "", "output": '{"name": "list_users", "arguments": {"keyword": "张伟"}}'},
    ]
    import json
    with open(_LLM_DIR / "door_assistant_samples.jsonl", "w", encoding="utf-8") as f:
        for s in llm_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    pipe.release()

    # 5) README
    (_TEACHING / "README.md").write_text(
        "# 教学数据包\n\n"
        "| 目录 | 内容 | 用途 |\n"
        "|---|---|---|\n"
        "| annotate_todo/ | 待标注图片（无标签） | 学生标注练习 |\n"
        "| annotate_ref/ | 5 张图片 + 模型预标注标签（YOLO 格式） | 标注示范与对照 |\n"
        "| yolo_dataset/ | 完整训练集（SCRFD 模型预标注，train/val 划分好，含 face.yaml） | 检测模型微调训练 |\n"
        "| llm_finetune/ | 门禁助手指令数据样例（Alpaca 格式 JSONL） | LLM LoRA 微调数据模板 |\n\n"
        "说明：\n"
        "1. 所有标签由本项目 SCRFD 检测模型**预标注**生成，是「模型预标注 + 人工校正」"
        "工业流程的教学素材——预标注可能有误差，学生应逐张检查校正（这也是标注练习的一部分）。\n"
        "2. 检测类别只有一类 `0: face`，无论人脸是谁。\n"
        "3. 训练前建议把 yolo_dataset 复制到**不含中文的路径**（如 `D:/datasets/face`），"
        "避免个别工具对中文路径不兼容。\n"
        "4. 重新生成：`python scripts/make_teaching_data.py`。\n",
        encoding="utf-8")

    print(f"\n完成：{_TEACHING}")
    print(f"  待标注 {len(records)} 张 / 参考标注 {_N_REF} 张 / "
          f"训练集 {len(records)} 张（val {n_val} 张）")


if __name__ == "__main__":
    main()
