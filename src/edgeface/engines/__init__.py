"""推理引擎包。

- base.py         FaceEngine 接口与 Detection 数据结构
- onnx_engine.py  PC/WSL 实现（insightface + onnxruntime）
- rknn_engine.py  RK3588 板端实现（rknnlite + NPU，兼容其它 RK 板）
"""
from .base import FaceEngine, Detection

__all__ = ["FaceEngine", "Detection"]
