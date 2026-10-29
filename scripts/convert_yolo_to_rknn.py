# -*- coding: utf-8 -*-
"""YOLO 微调模型（best.onnx）→ RKNN 转换脚本（教学微调路线配套）。

运行环境：Linux x86_64 / WSL2 + rknn-toolkit2（Windows 原生不可运行）。
⚠️ 本脚本未在板端实测（板端部署属教学文档第 07 章待办），
   转换参数按 rknn-toolkit2 官方 YOLO 示例编写；主检测链路仍用 SCRFD。

用法：
    python scripts/convert_yolo_to_rknn.py -i /path/to/best.onnx -p rk3588
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from rknn.api import RKNN
except ImportError:
    print("找不到 rknn.api，请在 rknn2 环境（Linux/WSL2，已装 rknn-toolkit2）里运行")
    sys.exit(1)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "assets" / "models"

# YOLO11 导出时图像归一化已内置在 onnx 图中（blobFromImage 风格），
# 转换时不再叠加 mean/std，避免重复归一化
MEAN = [[0, 0, 0]]
STD = [[1, 1, 1]]


def convert(onnx_path, output_path, platform, input_size):
    if not os.path.exists(onnx_path):
        print(f"找不到 onnx：{onnx_path}")
        sys.exit(1)

    print(f"ONNX  ：{onnx_path}")
    print(f"输出  ：{output_path}")
    print(f"平台  ：{platform}    输入：1x3x{input_size}x{input_size}（量化 FP16）")

    rknn = RKNN(verbose=False)
    print("\n--> config")
    rknn.config(
        mean_values=MEAN,
        std_values=STD,
        target_platform=platform,
        optimization_level=3,
    )

    print("--> load_onnx")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"  load_onnx 失败：{ret}")
        sys.exit(1)

    print("--> build（FP16）")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"  build 失败：{ret}")
        sys.exit(1)

    print("--> export_rknn")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"  export 失败：{ret}")
        sys.exit(1)

    rknn.release()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n✓ 完成：{output_path}  ({size_mb:.1f} MB)")
    print("注意：板端接入需自行实现 YOLO 输出的解码后处理（与 SCRFD 不同），")
    print("本脚本产出模型即可由 RKNNLite 加载推理。")


def main():
    parser = argparse.ArgumentParser(description="YOLO onnx → RKNN（FP16）")
    parser.add_argument("-i", "--onnx", required=True, help="输入 best.onnx 路径")
    parser.add_argument("-o", "--output", default=None, help="输出 .rknn 路径")
    parser.add_argument("-p", "--platform", default="rk3588",
                        choices=["rk3562", "rk3566", "rk3568", "rk3576", "rk3588"])
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    _DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or str(
        _DEFAULT_OUT_DIR / f"yolo11_face_fp16_{args.platform}.rknn")

    convert(args.onnx, output, args.platform, args.input_size)


if __name__ == "__main__":
    main()
