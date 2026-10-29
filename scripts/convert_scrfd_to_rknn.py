"""SCRFD-500m（人脸检测）ONNX → RKNN 转换脚本。

运行环境：Linux x86_64 / WSL2 + rknn-toolkit2（Windows 原生不可运行）。
转出的 .rknn 拷到板端由 rknn_engine 加载。

用法：
    python scripts/convert_scrfd_to_rknn.py                  # FP16（推荐）
    python scripts/convert_scrfd_to_rknn.py -q INT8 --calib-dir face_db_raw

模型来源：PC 跑 tools/register.py 后自动下载到
~/.insightface/models/buffalo_s/det_500m.onnx
"""
import os
import sys
import argparse
from pathlib import Path

try:
    from rknn.api import RKNN
except ImportError:
    print("找不到 rknn.api，请在 rknn2 环境（Linux/WSL2，已装 rknn-toolkit2）里运行")
    sys.exit(1)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ONNX = os.path.expanduser("~/.insightface/models/buffalo_s/det_500m.onnx")
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "assets" / "models"

# SCRFD 归一化参数（与 InsightFace 训练一致，不能改）。转换时内置进模型，
# 板端喂 [0,255] 原始像素即可，不能重复归一化
SCRFD_MEAN = [[127.5, 127.5, 127.5]]
SCRFD_STD = [[128.0, 128.0, 128.0]]
INPUT_NAME = "input.1"


def load_calib_dataset(calib_dir, max_n=300):
    """扫描校准目录，把图片路径写入 txt（rknn 的 dataset 参数要求）。"""
    files = []
    for root, _, fs in os.walk(calib_dir):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                files.append(os.path.join(root, f))
    files = sorted(files)[:max_n]
    if not files:
        return None
    list_file = os.path.join(calib_dir, "_calib_list.txt")
    with open(list_file, "w") as f:
        f.write("\n".join(os.path.abspath(p) for p in files) + "\n")
    return list_file


def convert(onnx_path, output_path, platform, quantize, input_size, calib_dir):
    if not os.path.exists(onnx_path):
        print(f"找不到 SCRFD onnx：{onnx_path}")
        print("先在 PC 跑一次 tools/register.py，insightface 会自动下载模型")
        sys.exit(1)

    print(f"ONNX  ：{onnx_path}")
    print(f"输出  ：{output_path}")
    print(f"平台  ：{platform}    量化：{quantize}    输入：1x3x{input_size}x{input_size}")

    rknn = RKNN(verbose=False)

    # 配置：mean/std 内置 + 编译优化等级 3
    print("\n--> config")
    rknn.config(
        mean_values=SCRFD_MEAN,
        std_values=SCRFD_STD,
        target_platform=platform,
        optimization_level=3,
    )

    # 加载 ONNX：动态输入锁定到固定尺寸
    print("--> load_onnx")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=[INPUT_NAME],
        input_size_list=[[1, 3, input_size, input_size]],
    )
    if ret != 0:
        print(f"  load_onnx 失败：{ret}")
        sys.exit(1)

    # 构建：FP16 直接构建；INT8 需校准集（建议 ≥200 张人脸图）
    print("--> build")
    if quantize == "INT8":
        if not calib_dir:
            print("  INT8 量化需要 --calib-dir 指向人脸图片目录")
            sys.exit(1)
        dataset_file = load_calib_dataset(calib_dir)
        if not dataset_file:
            print(f"  在 {calib_dir} 下没找到图片")
            sys.exit(1)
        print(f"  校准集：{dataset_file}")
        ret = rknn.build(do_quantization=True, dataset=dataset_file)
    else:
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


def main():
    parser = argparse.ArgumentParser(description="SCRFD-500m → RKNN")
    parser.add_argument("-i", "--onnx", default=_DEFAULT_ONNX, help="输入 onnx 路径")
    parser.add_argument("-o", "--output", default=None, help="输出 .rknn 路径")
    parser.add_argument("-p", "--platform", default="rk3566",
                        choices=["rk3562", "rk3566", "rk3568", "rk3576", "rk3588"])
    parser.add_argument("-q", "--quantize", default="FP16", choices=["FP16", "INT8"])
    parser.add_argument("--input-size", type=int, default=640,
                        help="检测输入尺寸（小脸多可 960，但更慢）")
    parser.add_argument("--calib-dir", default=None, help="INT8 校准图片目录")
    args = parser.parse_args()

    _DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or str(
        _DEFAULT_OUT_DIR / f"det_500m_{args.quantize.lower()}_{args.platform}.rknn")

    convert(args.onnx, output, args.platform, args.quantize, args.input_size, args.calib_dir)


if __name__ == "__main__":
    main()
