"""MobileFaceNet（w600k_mbf，人脸识别）ONNX → RKNN 转换脚本。

运行环境：Linux x86_64 / WSL2 + rknn-toolkit2。

关键参数：输入 112×112（与 FaceAligner 输出一致，不可改）；
FP16 推荐（识别模型对 INT8 量化敏感，LFW 精度可能掉 1~3%）。

用法：
    python scripts/convert_mbf_to_rknn.py
    python scripts/convert_mbf_to_rknn.py -q INT8 --calib-dir face_db_raw

模型来源：~/.insightface/models/buffalo_s/w600k_mbf.onnx
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
_DEFAULT_ONNX = os.path.expanduser("~/.insightface/models/buffalo_s/w600k_mbf.onnx")
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "assets" / "models"

# 识别模型归一化（与 InsightFace 训练一致，不能改）
MBF_MEAN = [[127.5, 127.5, 127.5]]
MBF_STD = [[127.5, 127.5, 127.5]]
INPUT_NAME = "input.1"
INPUT_SIZE = 112


def load_calib_dataset(calib_dir, max_n=300):
    """扫描校准目录，把图片路径写入 txt。"""
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


def convert(onnx_path, output_path, platform, quantize, calib_dir):
    if not os.path.exists(onnx_path):
        print(f"找不到识别模型 onnx：{onnx_path}")
        print("先在 PC 跑一次 tools/register.py，insightface 会自动下载模型")
        sys.exit(1)

    print(f"ONNX  ：{onnx_path}")
    print(f"输出  ：{output_path}")
    print(f"平台  ：{platform}    量化：{quantize}    输入：1x3x{INPUT_SIZE}x{INPUT_SIZE}")

    rknn = RKNN(verbose=False)

    print("\n--> config")
    rknn.config(
        mean_values=MBF_MEAN,
        std_values=MBF_STD,
        target_platform=platform,
        optimization_level=3,
    )

    # batch 维锁定到 1
    print("--> load_onnx")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=[INPUT_NAME],
        input_size_list=[[1, 3, INPUT_SIZE, INPUT_SIZE]],
    )
    if ret != 0:
        print(f"  load_onnx 失败：{ret}")
        sys.exit(1)

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
        print("  注：理想校准集是已对齐的 112×112 人脸；用原图 resize 精度可能略差")
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
    parser = argparse.ArgumentParser(description="w600k_mbf → RKNN")
    parser.add_argument("-i", "--onnx", default=_DEFAULT_ONNX, help="输入 onnx 路径")
    parser.add_argument("-o", "--output", default=None, help="输出 .rknn 路径")
    parser.add_argument("-p", "--platform", default="rk3566",
                        choices=["rk3562", "rk3566", "rk3568", "rk3576", "rk3588"])
    parser.add_argument("-q", "--quantize", default="FP16", choices=["FP16", "INT8"])
    parser.add_argument("--calib-dir", default=None, help="INT8 校准图片目录")
    args = parser.parse_args()

    _DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or str(
        _DEFAULT_OUT_DIR / f"w600k_mbf_{args.quantize.lower()}_{args.platform}.rknn")

    convert(args.onnx, output, args.platform, args.quantize, args.calib_dir)


if __name__ == "__main__":
    main()
