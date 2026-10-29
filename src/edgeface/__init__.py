"""edgeface 包入口：推理后端探测与引擎工厂。

用法：
    engine = create_engine()          # 板端自动选 RKNN，PC 自动选 ONNX
    pipe = FacePipeline(engine)
    results = pipe.process(frame_bgr)

板端模型路径：默认扫描 assets/models/ 下 det_500m_*.rknn /
w600k_mbf_*.rknn（优先文件名含当前板型，如 rk3588），也可用环境变量
EDGEFACE_DET_MODEL / EDGEFACE_REC_MODEL 显式指定。
"""
import os
from pathlib import Path

from .engines.base import FaceEngine, Detection
from .engines.onnx_engine import OnnxEngine
from .engines.rknn_engine import RknnEngine

_MODEL_DIR = Path(__file__).resolve().parents[2] / "assets" / "models"


def _detect_board() -> str:
    """从设备树读板型（如 "rk3588"）；读不到返回 None。"""
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            compat = f.read().decode("utf-8", errors="ignore").split("\x00")
        for c in compat:
            if c.startswith("rockchip,rk"):
                return c.replace("rockchip,", "").split(",")[0]
    except OSError:
        pass
    return None


def _find_rknn_model(pattern: str) -> str:
    """在 assets/models/ 下找第一个匹配的 .rknn；优先文件名含当前板型。"""
    env_name = "EDGEFACE_DET_MODEL" if pattern.startswith("det_") \
        else "EDGEFACE_REC_MODEL"
    if os.environ.get(env_name):
        return os.environ[env_name]
    if not _MODEL_DIR.is_dir():
        return None
    candidates = sorted(_MODEL_DIR.glob(pattern))
    if not candidates:
        return None
    board = _detect_board()
    if board:
        preferred = [p for p in candidates if board in p.name]
        if preferred:
            return str(preferred[0])
    return str(candidates[0])


def detect_backend() -> str:
    """探测当前环境可用后端，返回 "rknn" / "onnx" / None。

    判断顺序：能 import rknnlite.api → "rknn"（板端）；
    否则能 import insightface → "onnx"（PC/WSL）。
    """
    try:
        import rknnlite.api  # noqa: F401
        return "rknn"
    except ImportError:
        pass
    try:
        import insightface  # noqa: F401
        return "onnx"
    except ImportError:
        pass
    return None


# 进程内探测一次，UI/工具直接读该值
BACKEND = detect_backend()


def create_engine(det_model_path: str = None, rec_model_path: str = None,
                  input_size: int = 640) -> FaceEngine:
    """按当前环境创建引擎。

    Args:
        det_model_path: 检测 .rknn 路径；rknn 后端默认自动找
            assets/models/det_500m_*.rknn
        rec_model_path: 识别 .rknn 路径；rknn 后端默认自动找
            assets/models/w600k_mbf_*.rknn
        input_size: 检测输入边长；rknn 后端须与模型转换时一致

    Raises:
        RuntimeError: 两个后端都不可用
        FileNotFoundError: rknn 后端找不到模型文件（提示先跑转换脚本）
    """
    backend = detect_backend()

    if backend == "rknn":
        det_model_path = det_model_path or _find_rknn_model("det_500m_*.rknn")
        rec_model_path = rec_model_path or _find_rknn_model("w600k_mbf_*.rknn")
        missing = [p for p in (det_model_path, rec_model_path) if not p]
        if missing:
            raise FileNotFoundError(
                f"找不到板端 .rknn 模型：{missing}。请在 Linux/WSL2 下运行 "
                f"scripts/convert_scrfd_to_rknn.py 与 convert_mbf_to_rknn.py "
                f"（或设置 EDGEFACE_DET_MODEL / EDGEFACE_REC_MODEL）"
            )
        return RknnEngine(det_model_path, rec_model_path, input_size=input_size)

    if backend == "onnx":
        return OnnxEngine(det_size=(input_size, input_size))

    raise RuntimeError(
        "没有可用的推理后端。PC 安装 insightface+onnxruntime；"
        "板端安装 rknnlite-api。"
    )
