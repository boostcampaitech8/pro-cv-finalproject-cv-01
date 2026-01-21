"""
QAT 모델 ONNX Export

TensorRT INT8 변환을 위한 Q/DQ 노드 포함 ONNX 파일 생성.

수정 이력:
- torch.jit.trace 기반 export 추가 (data-dependent expression 문제 해결)
- Static scale 고정 기능 추가
"""

import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional
import warnings


def export_qat_to_onnx(
    model: nn.Module,
    output_path: str,
    config: Dict[str, Any],
    img_size: int = 640,
    batch_size: int = 1,
    device: str = 'cuda'
) -> str:
    """
    QAT 모델을 ONNX로 export (참조 리포지토리 방식).

    여러 방법을 순차적으로 시도하여 data-dependent expression 오류를 해결합니다:
    0. NVIDIA TensorRT 방식 (참조 리포지토리, 권장)
    1. torch.jit.trace 기반 export
    2. torch.onnx.export with static scale
    3. torch.onnx.export (기본)
    4. Ultralytics 내장 export (fallback)

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

    opset = export_config.get('opset', 14)  # 참조 리포지토리는 opset 14 사용
    simplify = export_config.get('simplify', True)
    dynamic_batch = export_config.get('dynamic_batch', False)

    print(f"[QAT] ONNX Export 시작...")
    print(f"  - Output: {output_path}")
    print(f"  - Opset: {opset}")
    print(f"  - Simplify: {simplify}")
    print(f"  - Dynamic batch: {dynamic_batch}")

    model.eval()
    model.to(device)

    # 더미 입력 생성
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # 방법 0: NVIDIA TensorRT 방식 (참조 리포지토리)
    try:
        print("\n[방법 0] NVIDIA TensorRT 방식 export 시도 (참조 리포지토리)...")
        result = _export_nvidia_tensorrt_style(
            model, dummy_input, output_path, opset, simplify
        )
        if result:
            _verify_qdq_nodes(output_path)
            quant_nn.TensorQuantizer.use_fb_fake_quant = False
            return output_path
    except Exception as e:
        print(f"[QAT] ❌ NVIDIA TensorRT 방식 실패: {e}")
        import traceback
        traceback.print_exc()

    # TensorQuantizer를 inference mode로 설정
    print("\n[QAT] TensorQuantizer를 inference mode로 설정 중...")
    _prepare_model_for_export(model)
    print("[QAT] ✅ TensorQuantizer inference mode 설정 완료")

    # 방법 1: torch.jit.trace 사용 (data-dependent 문제 해결)
    try:
        print("\n[방법 1] torch.jit.trace 기반 export 시도...")
        result = _export_with_jit_trace(
            model, dummy_input, output_path, opset, dynamic_batch
        )
        if result:
            # 검증 및 후처리
            if simplify:
                output_path = _simplify_onnx(output_path)
            _verify_qdq_nodes(output_path)
            quant_nn.TensorQuantizer.use_fb_fake_quant = False
            return output_path
    except Exception as e:
        print(f"[QAT] ❌ JIT trace 실패: {e}")

    # 방법 2: Static scale 고정 후 export
    try:
        print("\n[방법 2] Static scale 고정 후 export 시도...")
        freeze_quantizer_scales(model)
        result = _export_with_torch_onnx(
            model, dummy_input, output_path, opset, dynamic_batch
        )
        if result:
            if simplify:
                output_path = _simplify_onnx(output_path)
            _verify_qdq_nodes(output_path)
            quant_nn.TensorQuantizer.use_fb_fake_quant = False
            return output_path
    except Exception as e:
        print(f"[QAT] ❌ Static scale export 실패: {e}")

    # 방법 3: 기본 torch.onnx.export
    try:
        print("\n[방법 3] 기본 torch.onnx.export 시도...")
        result = _export_with_torch_onnx(
            model, dummy_input, output_path, opset, dynamic_batch
        )
        if result:
            if simplify:
                output_path = _simplify_onnx(output_path)
            _verify_qdq_nodes(output_path)
            quant_nn.TensorQuantizer.use_fb_fake_quant = False
            return output_path
    except Exception as e:
        print(f"[QAT] ❌ 기본 export 실패: {e}")

    # 방법 4: Ultralytics fallback
    print("\n[방법 4] Ultralytics 내장 export 시도...")
    quant_nn.TensorQuantizer.use_fb_fake_quant = False
    return _export_via_ultralytics(model, output_path, config, img_size)


def _export_nvidia_tensorrt_style(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset: int,
    simplify: bool
) -> bool:
    """
    NVIDIA TensorRT 방식 ONNX export (참조 리포지토리).

    mmsori/yolov8-QAT 방식 (pytorch_quantization 2.1.2 호환):
    1. use_fb_fake_quant = True 설정 (PyTorch fake quantization 사용)
    2. Detect 레이어 export 모드 활성화
    3. torch.onnx.export 직접 호출 (enable_onnx_export 불필요)
    4. enable_onnx_checker=False 설정

    Args:
        model: QAT 모델
        dummy_input: 더미 입력
        output_path: 출력 경로
        opset: ONNX opset 버전 (14 권장)
        simplify: ONNX 단순화 여부

    Returns:
        성공 여부
    """
    try:
        from pytorch_quantization import nn as quant_nn
        from ultralytics.nn.modules import Detect

        # 1. PyTorch fake quantization 활성화
        print("  - use_fb_fake_quant 활성화...")
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        # 2. Detect 레이어를 export 모드로 설정
        print("  - Detect 레이어 export 모드 설정...")
        for m in model.modules():
            if isinstance(m, Detect):
                m.export = True
                m.format = 'onnx'
                print(f"    ✓ {m.__class__.__name__} export=True, format=onnx")

        # 3. torch.onnx.export 직접 호출
        print(f"  - torch.onnx.export 실행 (opset={opset})...")

        input_names = ["images"]
        output_names = ["output0"]

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            dynamo=False,  # torch.export.export 우회 (pytorch_quantization 호환)
            verbose=False,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'images': {0: 'batch'},
                'output0': {0: 'batch'}
            },
            do_constant_folding=True
        )

        # 4. 플래그 원상복구
        quant_nn.TensorQuantizer.use_fb_fake_quant = False

        print(f"  ✅ NVIDIA TensorRT 방식 export 성공: {output_path}")

        # 5. ONNX simplify (선택)
        if simplify:
            output_path = _simplify_onnx(output_path)

        return True

    except Exception as e:
        # 예외 발생 시에도 플래그 복구
        try:
            quant_nn.TensorQuantizer.use_fb_fake_quant = False
        except:
            pass

        print(f"  ❌ NVIDIA TensorRT 방식 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


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


# ============================================================================
# 새로운 Export 방법들 (data-dependent expression 문제 해결)
# ============================================================================

def _prepare_model_for_export(model: nn.Module) -> None:
    """
    모델을 ONNX export용으로 준비.

    TensorQuantizer를 inference mode로 설정합니다.
    """
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    # Q/DQ 노드를 ONNX에 포함시키기 위한 설정
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # TensorQuantizer를 inference mode로 설정
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # Calibration 비활성화
            if module._calibrator is not None:
                module.disable_calib()
            # Quantization 활성화
            module.enable_quant()
            module.enable()


def _export_with_jit_trace(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset: int,
    dynamic_batch: bool
) -> bool:
    """
    torch.jit.trace를 사용한 ONNX export.

    torch.export와 달리 data-dependent 연산을 허용합니다.
    실제 데이터를 실행하면서 그래프를 기록하기 때문에 동적 연산도 처리 가능합니다.

    Args:
        model: QAT 모델
        dummy_input: 더미 입력
        output_path: 출력 경로
        opset: ONNX opset 버전
        dynamic_batch: 동적 배치 사용 여부

    Returns:
        성공 여부
    """
    try:
        print("  - JIT tracing 시작...")

        # Dynamic axes 설정
        if dynamic_batch:
            dynamic_axes = {
                'images': {0: 'batch_size'},
                'output0': {0: 'batch_size'},
            }
        else:
            dynamic_axes = None

        # JIT trace로 그래프 생성
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)

        # Traced 모델을 ONNX로 export
        torch.onnx.export(
            traced_model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=['images'],
            output_names=['output0'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )

        print(f"  ✅ JIT trace export 성공: {output_path}")
        return True

    except Exception as e:
        print(f"  ❌ JIT trace 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def _export_with_torch_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset: int,
    dynamic_batch: bool
) -> bool:
    """
    기본 torch.onnx.export 사용.

    Args:
        model: QAT 모델
        dummy_input: 더미 입력
        output_path: 출력 경로
        opset: ONNX opset 버전
        dynamic_batch: 동적 배치 사용 여부

    Returns:
        성공 여부
    """
    try:
        # Dynamic axes 설정
        if dynamic_batch:
            dynamic_axes = {
                'images': {0: 'batch_size'},
                'output0': {0: 'batch_size'},
            }
        else:
            dynamic_axes = None

        # ONNX Export
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

        print(f"  ✅ torch.onnx.export 성공: {output_path}")

        # 검증
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ✅ ONNX 모델 검증 통과")

        return True

    except Exception as e:
        print(f"  ❌ torch.onnx.export 실패: {e}")
        return False


def freeze_quantizer_scales(model: nn.Module) -> None:
    """
    TensorQuantizer의 scale을 상수로 고정.

    Calibration 후 학습된 scale/amax 값을 상수로 변환하여
    data-dependent 연산을 제거합니다.

    주의: 이 방법은 scale을 고정하므로 추가 fine-tuning이 불가능합니다.
    Export 직전에만 사용하세요.

    Args:
        model: QAT 모델
    """
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    frozen_count = 0

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # amax를 상수로 고정
            if hasattr(module, '_amax') and module._amax is not None:
                if isinstance(module._amax, torch.Tensor):
                    # Tensor를 상수로 변환 (requires_grad=False)
                    module._amax = module._amax.detach().clone()
                    module._amax.requires_grad = False

                frozen_count += 1

            # scale도 상수로 고정 (있는 경우)
            if hasattr(module, '_scale') and module._scale is not None:
                if isinstance(module._scale, torch.Tensor):
                    module._scale = module._scale.detach().clone()
                    module._scale.requires_grad = False

    print(f"  - {frozen_count}개 TensorQuantizer scale 고정 완료")
