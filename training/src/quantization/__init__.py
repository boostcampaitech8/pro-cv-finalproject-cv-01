"""
QAT (Quantization-Aware Training) 패키지

NVIDIA pytorch-quantization을 사용한 YOLOv8s QAT 구현.
TensorRT INT8 최적 변환을 위한 Q/DQ 노드 삽입 지원.
"""

from .qat_utils import (
    initialize_quantization,
    prepare_model_for_qat,
    collect_calibration_stats,
    disable_quantization,
    enable_quantization,
)
from .export_qat import export_qat_to_onnx

__all__ = [
    "initialize_quantization",
    "prepare_model_for_qat",
    "collect_calibration_stats",
    "disable_quantization",
    "enable_quantization",
    "export_qat_to_onnx",
]
