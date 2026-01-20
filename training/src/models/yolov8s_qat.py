"""
YOLOv8s QAT 모델 래퍼

config_qat.yaml에서 model_module: "yolov8s_qat"로 설정하면 이 모듈이 로드됩니다.
Pre-trained 모델을 로드하고 QAT용으로 변환합니다.
"""

import os
from ultralytics import YOLO
from typing import Dict, Any


def get_model(config: Dict[str, Any]) -> YOLO:
    """
    QAT용 YOLOv8s 모델을 로드합니다.

    1. Pre-trained best.pt 로드 (일반 학습 결과)
    2. pytorch-quantization 초기화 (모델 로드 전에 수행해야 함)
    3. 양자화 모듈 교체

    Args:
        config: config_qat.yaml에서 로드된 설정 딕셔너리

    Returns:
        YOLO: QAT용으로 준비된 모델
    """
    qat_config = config.get('qat', {})
    pretrained_path = qat_config.get('pretrained_path', '')

    # pretrained_path 절대 경로로 변환
    if pretrained_path and not os.path.isabs(pretrained_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pretrained_path = os.path.abspath(os.path.join(base_dir, pretrained_path))

    # Pre-trained 모델 존재 확인
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pre-trained 모델을 찾을 수 없습니다: {pretrained_path}\n"
            "먼저 일반 학습을 수행하세요: uv run python run_exp.py --config config.yaml"
        )

    print(f"[QAT] Pre-trained 모델 로드: {pretrained_path}")

    # 모델 로드 (먼저 일반 모델로 로드)
    model = YOLO(pretrained_path)

    print(f"[QAT] 모델 로드 완료")
    print(f"  - Classes: {len(model.names)}")
    print(f"  - Names: {list(model.names.values())}")

    # Conv2d → QuantConv2d 수동 교체 (Ultralytics YOLO는 이미 만들어진 모델을 로드하므로
    # quant_modules.initialize()가 효과 없음. 따라서 수동으로 교체해야 함)
    print("\n" + "="*60)
    print("[QAT] Conv2d → QuantConv2d 교체 시작...")
    print("="*60)

    try:
        from src.quantization import replace_conv_with_quantconv, disable_detect_head_quantization

        # 수동으로 Conv2d를 QuantConv2d로 교체
        print("[QAT] replace_conv_with_quantconv() 호출 중...")
        replace_conv_with_quantconv(model.model, config)

        # Detect Head 양자화 비활성화 (정확도 확보)
        print("[QAT] Detect Head 양자화 비활성화 중...")
        disable_detect_head_quantization(model.model)

        # 교체 확인
        from pytorch_quantization import nn as quant_nn
        quantizer_count = sum(1 for m in model.model.modules() if isinstance(m, quant_nn.TensorQuantizer))
        print(f"\n[QAT] ✅ 양자화 교체 완료!")
        print(f"  - TensorQuantizer 개수: {quantizer_count}")

        if quantizer_count == 0:
            print(f"\n[QAT] ⚠️ 경고: TensorQuantizer가 생성되지 않았습니다!")
            print(f"  이것은 심각한 문제입니다. QAT가 적용되지 않을 것입니다.")
            raise RuntimeError("QAT 모듈 교체 실패: TensorQuantizer가 없음")

        print("="*60 + "\n")

    except ImportError as e:
        print(f"\n[QAT] ❌ ImportError: {e}")
        print(f"  pytorch-quantization을 찾을 수 없습니다.")
        print(f"  설치: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")
        print(f"  QAT 없이 일반 fine-tuning으로 진행합니다.")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n[QAT] ❌ 양자화 변환 오류: {e}")
        print(f"  이것은 심각한 문제입니다. 다음을 확인하세요:")
        print(f"    1. pytorch-quantization 버전")
        print(f"    2. PyTorch 버전 호환성")
        print(f"    3. config_qat.yaml 설정")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        raise  # 에러를 다시 raise해서 학습 중단

    return model


def _initialize_qat_if_available(config: Dict[str, Any]) -> bool:
    """
    pytorch-quantization이 설치된 경우 QAT 초기화 수행.

    Returns:
        초기화 성공 여부
    """
    try:
        from src.quantization import initialize_quantization, prepare_model_for_qat
        initialize_quantization(config)
        return True
    except ImportError as e:
        print(f"[QAT] 경고: pytorch-quantization을 찾을 수 없습니다.")
        print(f"  설치: pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")
        print(f"  QAT 없이 일반 fine-tuning으로 진행합니다.")
        return False
    except Exception as e:
        print(f"[QAT] 초기화 오류: {e}")
        return False


def get_model_for_calibration(config: Dict[str, Any]) -> YOLO:
    """
    Calibration용 모델 로드.

    Calibration 단계에서는 학습 없이 forward pass만 수행합니다.

    Args:
        config: QAT 설정

    Returns:
        Calibration용 모델
    """
    model = get_model(config)
    model.eval()
    return model


def prepare_for_qat_training(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    QAT 학습을 위한 모델 준비.

    Calibration 후 호출하여 fine-tuning 준비를 합니다.

    Args:
        model: Calibration 완료된 모델
        config: QAT 설정

    Returns:
        QAT 학습 준비된 모델
    """
    try:
        from src.quantization import enable_quantization
        enable_quantization(model.model)
        print("[QAT] 양자화 활성화 완료")
    except ImportError:
        pass

    return model
