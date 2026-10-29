# -*- coding: utf-8 -*-
"""SCRFD 自定义后处理验证：板端唯一使用、PC 端不执行的代码路径。

用 onnxruntime 直接跑 det_500m.onnx 拿 9 个输出张量 → 喂给
edgeface.postprocess.detect → 与 insightface 官方检测结果对比。
对比通过 = 板端 rknn 引擎的后处理逻辑正确（rknn 模型与 onnx 同源）。

用法：python diagnostics/scrfd_postprocess_test.py [图片路径...]
不传参数则自动用 face_db_raw 的注册照 + testvideo 抽帧。
"""
import glob
import os
import sys
import warnings

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.path.join(_PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import onnxruntime as ort  # noqa: E402

from edgeface.postprocess import detect as pp_detect  # noqa: E402
from edgeface import io as edge_io  # noqa: E402（中文路径安全读写）

warnings.filterwarnings("ignore")

_ONNX = os.path.expanduser("~/.insightface/models/buffalo_s/det_500m.onnx")
INPUT_SIZE = 640
INPUT_NAME = "input.1"
MEAN, STD = 127.5, 128.0


def _preprocess(img_bgr):
    """与板端 rknn_engine.detect 完全一致的几何：等比缩放 + 左上对齐黑边填充。
    布局差异：onnx 为 NCHW + 显式归一化；rknn 为 NHWC 无归一化
    （转换时已内置）—— 两者等价，正是要验证的点。
    Returns: (blob, new_w, new_h, det_scale)
    """
    h, w = img_bgr.shape[:2]
    im_ratio = h / float(w)
    if im_ratio > 1.0:
        new_h = INPUT_SIZE
        new_w = max(1, int(new_h / im_ratio))
    else:
        new_w = INPUT_SIZE
        new_h = max(1, int(new_w * im_ratio))
    det_scale = new_h / float(h)
    resized = cv2.resize(img_bgr, (new_w, new_h))
    det_img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = resized
    rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
    blob = rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    return (blob - MEAN) / STD, new_w, new_h, det_scale


def _insightface_detect(img_bgr):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(INPUT_SIZE, INPUT_SIZE))
    faces = app.get(img_bgr)
    return [{"bbox": f.bbox.astype(np.float32),
             "kps": f.kps.astype(np.float32),
             "score": float(f.det_score)} for f in faces]


def _match_result(refs, ours, img_hw, iou_thresh=0.6):
    """每个参考框找 IoU 最高的我方框，返回 (ok, 详情)。"""
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

    lines = []
    matched_idx = set()
    all_ok = True
    for i, ref in enumerate(refs):
        best, best_iou = None, 0.0
        for j, o in enumerate(ours):
            if j in matched_idx:
                continue
            v = iou(ref["bbox"], o["bbox"])
            if v > best_iou:
                best, best_iou = j, v
        if best is None or best_iou < iou_thresh:
            all_ok = False
            lines.append(f"  ✗ 官方框 {i} (score={ref['score']:.3f}) 无对应我方框")
            continue
        matched_idx.add(best)
        o = ours[best]
        # 关键点平均误差（像素）
        kps_err = np.abs(ref["kps"] - o["kps"]).mean()
        lines.append(f"  ✓ 框 {i}: IoU={best_iou:.3f} 官方score={ref['score']:.3f} "
                     f"我方score={o['score']:.3f} 关键点误差={kps_err:.1f}px")
    if len(ours) > len(refs):
        all_ok = False
        lines.append(f"  ✗ 我方多出 {len(ours) - len(refs)} 个框")
    return all_ok, lines


def main():
    if not os.path.isfile(_ONNX):
        print(f"找不到 {_ONNX}，先跑一次 tools/register.py 让 insightface 下载模型")
        sys.exit(1)

    images = sys.argv[1:]
    if not images:
        raw_root = os.path.join(os.path.dirname(_PROJECT_ROOT), "face_db_raw")
        images = glob.glob(os.path.join(raw_root, "*", "*.jpg"))
        images += glob.glob(os.path.join(raw_root, "*", "*.png"))
        # 从测试视频抽 3 帧（真实场景）
        vids = glob.glob(os.path.join(_PROJECT_ROOT, "testvideo", "*.mp4"))
        if vids:
            cap = cv2.VideoCapture(vids[0])
            for fno in (60, 300, 600):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                ok, frame = cap.read()
                if ok:
                    path = os.path.join(_PROJECT_ROOT, "diagnostics",
                                        f"_vid_frame_{fno}.jpg")
                    edge_io.imwrite(path, frame)
                    images.append(path)
            cap.release()
    images = sorted(images)

    if not images:
        print("没有测试图片")
        sys.exit(1)

    sess = ort.InferenceSession(_ONNX, providers=["CPUExecutionProvider"])
    n_ok = 0
    print(f"测试图片 {len(images)} 张，输入 {INPUT_SIZE}x{INPUT_SIZE}\n")
    for path in images:
        img = edge_io.imread(path)
        if img is None:
            print(f"[跳过] 读不了 {path}")
            continue
        h, w = img.shape[:2]

        # 我方路径：raw onnx 输出 → 自定义后处理（与板端 rknn 引擎同几何）
        blob, new_w, new_h, det_scale = _preprocess(img)
        outputs = sess.run(None, {INPUT_NAME: blob})
        ours = pp_detect(outputs, INPUT_SIZE, det_thresh=0.5, nms_iou=0.4,
                         img_center_xy=(new_w / 2, new_h / 2))
        for r in ours:
            r["bbox"] /= det_scale
            r["kps"] /= det_scale

        # 官方路径：insightface 整流程
        refs = _insightface_detect(img)

        name = os.path.basename(path)
        ok, lines = _match_result(refs, ours, (h, w))
        n_ok += ok
        print(f"{name} ({w}x{h}): 官方 {len(refs)} 框 / 我方 {len(ours)} 框 "
              f"{'✅' if ok else '⚠️'}")
        for line in lines:
            print(line)
        print()

    print(f"===== 汇总: {n_ok}/{len(images)} 张图片结果一致 =====")


if __name__ == "__main__":
    main()
