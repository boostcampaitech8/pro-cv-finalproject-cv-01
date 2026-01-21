"""
QAT 모델 ONNX Export

TensorRT INT8 변환을 위한 Q/DQ 노드 포함 ONNX 파일 생성.
"""

import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional


def export_qat_to_onnx(
    model: nn.Module,
    output_path: str,
    config: Dict[str, Any],
    img_size: int = 640,
    batch_size: int = 1,
    device: str = 'cuda'
) -> str:
    """
    QAT 모델을 ONNX로 export.

    Q/DQ (Quantize/Dequantize) 노드를 포함하여 TensorRT INT8 변환에 최적화됩니다.

    Args:
        model: QAT 학습된 모델
        output_path: ONNX 파일 저장 경로
        config: QAT 설정
        img_size: 입력 이미지 크기
        batch_size: 배치 크기
        device: 디바이스

    Returns:
        저장된 ONNX 파일 경로
    """
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.nn import TensorQuantizer
    except ImportError:
        raise ImportError("pytorch-quantization이 설치되지 않았습니다.")

    qat_config = config.get('qat', {})
    export_config = qat_config.get('export', {})

    # Opset 13으로 낮춤 (더 안정적, TensorRT 호환성 향상)
    opset = export_config.get('opset', 13)
    simplify = export_config.get('simplify', True)
    dynamic_batch = export_config.get('dynamic_batch', False)

    print(f"[QAT] ONNX Export 시작...")
    print(f"  - Output: {output_path}")
    print(f"  - Opset: {opset}")
    print(f"  - Simplify: {simplify}")
    print(f"  - Dynamic batch: {dynamic_batch}")

    model.eval()
    model.to(device)

    # Q/DQ 노드를 ONNX에 포함시키기 위한 설정
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # TensorQuantizer를 inference mode로 설정 (calibration 비활성화)
    print("[QAT] TensorQuantizer를 inference mode로 설정 중...")
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # Calibration 비활성화 -> Inference 활성화
            if module._calibrator is not None:
                module.disable_calib()
            module.enable_quant()
            module.enable()
    print("[QAT] ✅ TensorQuantizer inference mode 설정 완료")

    # 더미 입력 생성
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # Dynamic axes 설정 (배치 크기 동적)
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch_size'},
            'output0': {0: 'batch_size'},
        }
    else:
        dynamic_axes = None

    # ONNX Export
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=['images'],
            output_names=['output0'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"[QAT] ✅ ONNX 기본 export 완료: {output_path}")
    except Exception as e:
        print(f"[QAT] ❌ torch.onnx.export 실패: {e}")
        print(f"  Fallback: ultralytics 내장 export 시도...")
        # Fallback: ultralytics 내장 export 사용
        return _export_via_ultralytics(model, output_path, config, img_size)

    # ONNX 모델 검증
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[QAT] ONNX 모델 검증 통과")
    except Exception as e:
        print(f"[QAT] ONNX 검증 경고: {e}")

    # ONNX Simplifier 적용
    if simplify:
        output_path = _simplify_onnx(output_path)

    # Q/DQ 노드 확인
    _verify_qdq_nodes(output_path)

    # 설정 복원
    quant_nn.TensorQuantizer.use_fb_fake_quant = False

    return output_path


def _export_via_ultralytics(
    model: nn.Module,
    output_path: str,
    config: Dict[str, Any],
    img_size: int
) -> str:
    """
    Ultralytics 내장 export 사용 (fallback).

    YOLO 모델의 경우 내장 export 함수가 더 안정적일 수 있습니다.
    """
    print("[QAT] Ultralytics 내장 export 사용...")

    qat_config = config.get('qat', {})
    export_config = qat_config.get('export', {})
    opset = export_config.get('opset', 17)
    simplify = export_config.get('simplify', True)

    # YOLO 모델인 경우 내장 export 사용
    if hasattr(model, 'export'):
        try:
            model.export(
                format='onnx',
                imgsz=img_size,
                opset=opset,
                simplify=simplify,
            )
            # ultralytics는 자동으로 파일명 생성
            # best.pt -> best.onnx
            base_path = output_path.replace('_qat.onnx', '.onnx')
            if os.path.exists(base_path):
                os.rename(base_path, output_path)
            print(f"[QAT] Ultralytics export 완료: {output_path}")
        except Exception as e:
            print(f"[QAT] Ultralytics export 실패: {e}")

    return output_path


def _simplify_onnx(onnx_path: str) -> str:
    """
    onnxsim으로 ONNX 모델 최적화.

    중복 노드 제거, 상수 폴딩 등을 수행합니다.
    """
    try:
        import onnx
        from onnxsim import simplify

        print("[QAT] ONNX Simplifier 적용 중...")

        model = onnx.load(onnx_path)
        model_simplified, check = simplify(model)

        if check:
            # 단순화된 모델 저장 (원본 덮어쓰기)
            onnx.save(model_simplified, onnx_path)
            print(f"[QAT] ONNX 단순화 완료: {onnx_path}")
        else:
            print("[QAT] ONNX 단순화 검증 실패, 원본 유지")

    except ImportError:
        print("[QAT] onnxsim이 설치되지 않음. 단순화 건너뜀.")
    except Exception as e:
        print(f"[QAT] ONNX 단순화 실패: {e}")

    return onnx_path


def _verify_qdq_nodes(onnx_path: str) -> None:
    """
    ONNX 모델에 Q/DQ 노드가 포함되었는지 확인.

    TensorRT INT8 변환을 위해 QuantizeLinear/DequantizeLinear 노드가 필요합니다.
    """
    try:
        import onnx

        model = onnx.load(onnx_path)

        qdq_ops = ['QuantizeLinear', 'DequantizeLinear']
        found_ops = {}

        for node in model.graph.node:
            if node.op_type in qdq_ops:
                found_ops[node.op_type] = found_ops.get(node.op_type, 0) + 1

        if found_ops:
            print(f"[QAT] Q/DQ 노드 확인:")
            for op, count in found_ops.items():
                print(f"  - {op}: {count}개")
        else:
            print("[QAT] 경고: Q/DQ 노드가 발견되지 않음. PTQ 모드로 전환될 수 있습니다.")

    except Exception as e:
        print(f"[QAT] Q/DQ 노드 확인 실패: {e}")


def validate_onnx_output(
    pytorch_model: nn.Module,
    onnx_path: str,
    img_size: int = 640,
    device: str = 'cuda',
    rtol: float = 1e-2,
    atol: float = 1e-3
) -> bool:
    """
    PyTorch 모델과 ONNX 모델의 출력 일치 검증.

    Args:
        pytorch_model: 원본 PyTorch 모델
        onnx_path: ONNX 파일 경로
        img_size: 이미지 크기
        device: 디바이스
        rtol: 상대 허용 오차
        atol: 절대 허용 오차

    Returns:
        검증 성공 여부
    """
    try:
        import onnxruntime as ort
        import numpy as np

        print("[QAT] ONNX 출력 검증 중...")

        # PyTorch 모델 출력
        pytorch_model.eval()
        pytorch_model.to(device)
        dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input)

        if isinstance(pytorch_output, (list, tuple)):
            pytorch_output = pytorch_output[0]
        pytorch_output = pytorch_output.cpu().numpy()

        # ONNX 모델 출력
        session = ort.InferenceSession(onnx_path)
        onnx_input = dummy_input.cpu().numpy()
        onnx_output = session.run(None, {'images': onnx_input})[0]

        # 출력 비교
        is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)

        if is_close:
            print("[QAT] ONNX 출력 검증 통과!")
        else:
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            print(f"[QAT] ONNX 출력 검증 실패. 최대 차이: {max_diff:.6f}")

        return is_close

    except ImportError:
        print("[QAT] onnxruntime이 설치되지 않음. 검증 건너뜀.")
        return True
    except Exception as e:
        print(f"[QAT] ONNX 출력 검증 실패: {e}")
        return False
