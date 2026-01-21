"""
QAT 유틸리티 함수

NVIDIA pytorch-quantization을 사용한 양자화 관련 유틸리티.
- Q/DQ 노드 삽입
- Calibration 수행
- 양자화 활성화/비활성화
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tqdm import tqdm


def initialize_quantization(config: Dict[str, Any]) -> None:
    """
    pytorch-quantization 라이브러리 초기화.

    QuantDescriptor를 설정하여 양자화 기본값을 지정합니다.
    반드시 모델 로드 전에 호출해야 합니다.

    Args:
        config: QAT 설정 (config_qat.yaml의 qat 섹션)
    """
    try:
        from pytorch_quantization import quant_modules
        from pytorch_quantization.tensor_quant import QuantDescriptor
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        raise ImportError(
            "pytorch-quantization이 설치되지 않았습니다. "
            "'pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com' "
            "으로 설치하세요."
        )

    qat_config = config.get('qat', {})
    quant_config = qat_config.get('quantization', {})
    calibration_config = qat_config.get('calibration', {})
    
    # 1. Quantization Modules 초기화 (Conv2d -> QuantConv2d 자동 교체)
    # 반드시 모델 로드 전에 수행되어야 함
    quant_modules.initialize()

    num_bits = quant_config.get('num_bits', 8)
    weight_per_channel = quant_config.get('weight_per_channel', True)
    calib_method = calibration_config.get('method', 'histogram')

    # Input activation quantization descriptor
    # histogram: 더 정확한 calibration (권장)
    # max: 빠르지만 덜 정확
    if calib_method == 'histogram':
        input_desc = QuantDescriptor(
            num_bits=num_bits,
            calib_method='histogram'
        )
    else:
        input_desc = QuantDescriptor(
            num_bits=num_bits,
            calib_method='max'
        )

    # Weight quantization descriptor
    # per_channel_axis=0: output channel별 양자화 (더 정확)
    if weight_per_channel:
        weight_desc = QuantDescriptor(
            num_bits=num_bits,
            axis=(0,)  # per-channel quantization
        )
    else:
        weight_desc = QuantDescriptor(num_bits=num_bits)

    # 기본 QuantDescriptor 설정
    quant_nn.QuantConv2d.set_default_quant_desc_input(input_desc)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(weight_desc)
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)

    print(f"[QAT] 양자화 초기화 완료:")
    print(f"  - quant_modules.initialize() 완료 (Conv2d → QuantConv2d)")
    print(f"  - Bits: {num_bits}")
    print(f"  - Weight per-channel: {weight_per_channel}")
    print(f"  - Calibration method: {calib_method}")


def prepare_model_for_qat(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    모델을 QAT용으로 준비.

    Conv2d → QuantConv2d, Linear → QuantLinear로 교체합니다.

    Args:
        model: 원본 PyTorch 모델
        config: QAT 설정

    Returns:
        QAT용으로 변환된 모델
    """
    try:
        from pytorch_quantization import quant_modules
    except ImportError:
        raise ImportError("pytorch-quantization이 설치되지 않았습니다.")

    qat_config = config.get('qat', {})
    quant_config = qat_config.get('quantization', {})

    quant_conv = quant_config.get('quant_conv', True)
    quant_linear = quant_config.get('quant_linear', True)
    skip_last_layers = quant_config.get('skip_last_layers', True)

    # quant_modules.initialize()를 사용하면 전체 모듈을 자동 교체
    # 하지만 더 세밀한 제어를 위해 수동 교체도 가능
    quant_modules.initialize()

    print(f"[QAT] 모델 양자화 준비 완료:")
    print(f"  - Quantize Conv2d: {quant_conv}")
    print(f"  - Quantize Linear: {quant_linear}")
    print(f"  - Skip last layers: {skip_last_layers}")

    # Detect Head의 마지막 레이어는 양자화 제외 (정확도 민감)
    if skip_last_layers:
        _disable_detect_head_quantization(model)

    return model


def replace_conv_with_quantconv(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    이미 로드된 모델의 Conv2d를 QuantConv2d로 수동 교체.

    Ultralytics YOLO는 체크포인트에서 모델을 복원할 때 이미 만들어진 모듈을 사용하므로
    quant_modules.initialize()가 효과가 없습니다. 이 함수는 로드된 모델의
    Conv2d를 직접 QuantConv2d로 교체합니다.

    Args:
        model: 이미 로드된 PyTorch 모델
        config: QAT 설정

    Returns:
        Conv2d가 QuantConv2d로 교체된 모델
    """
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.tensor_quant import QuantDescriptor
    except ImportError:
        raise ImportError("pytorch-quantization이 설치되지 않았습니다.")

    qat_config = config.get('qat', {})
    quant_config = qat_config.get('quantization', {})
    calibration_config = qat_config.get('calibration', {})

    num_bits = quant_config.get('num_bits', 8)
    weight_per_channel = quant_config.get('weight_per_channel', True)
    calib_method = calibration_config.get('method', 'histogram')

    # QuantDescriptor 설정
    if calib_method == 'histogram':
        input_desc = QuantDescriptor(num_bits=num_bits, calib_method='histogram')
    else:
        input_desc = QuantDescriptor(num_bits=num_bits, calib_method='max')

    if weight_per_channel:
        weight_desc = QuantDescriptor(num_bits=num_bits, axis=(0,))
    else:
        weight_desc = QuantDescriptor(num_bits=num_bits)

    replaced_count = 0

    # 모든 모듈을 순회하며 Conv2d를 QuantConv2d로 교체
    def replace_module(parent: nn.Module, name: str, module: nn.Module):
        nonlocal replaced_count

        # 이미 QuantConv2d인 경우 스킵
        if isinstance(module, quant_nn.QuantConv2d):
            return

        # Conv2d인 경우 QuantConv2d로 교체
        if isinstance(module, nn.Conv2d):
            # QuantConv2d 생성 (동일한 파라미터)
            quant_conv = quant_nn.QuantConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                quant_desc_input=input_desc,
                quant_desc_weight=weight_desc,
            )

            # 기존 가중치 복사
            quant_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                quant_conv.bias.data.copy_(module.bias.data)

            # 부모 모듈에서 교체
            setattr(parent, name, quant_conv)
            replaced_count += 1

    # 재귀적으로 모든 모듈 순회
    def recursive_replace(parent: nn.Module):
        for name, child in parent.named_children():
            # 먼저 자식의 자식들을 처리
            recursive_replace(child)
            # 그 다음 현재 자식을 교체
            replace_module(parent, name, child)

    recursive_replace(model)

    print(f"[QAT] Conv2d → QuantConv2d 수동 교체 완료:")
    print(f"  - 교체된 레이어 수: {replaced_count}")
    print(f"  - Bits: {num_bits}")
    print(f"  - Weight per-channel: {weight_per_channel}")

    return model


def disable_detect_head_quantization(model: nn.Module) -> None:
    """
    Detect Head의 마지막 레이어 양자화 비활성화.

    YOLOv8의 Detect Head는 정확도에 민감하므로 양자화를 제외합니다.
    """
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    # YOLOv8의 Detect 모듈 찾기
    # 모델 구조: model.model[-1]이 Detect 헤드
    for name, module in model.named_modules():
        # Detect 헤드의 cv2, cv3 레이어 (classification & box regression)
        if 'detect' in name.lower() or 'head' in name.lower():
            if isinstance(module, quant_nn.QuantConv2d):
                # 양자화 비활성화
                if hasattr(module, '_input_quantizer'):
                    module._input_quantizer.disable()
                if hasattr(module, '_weight_quantizer'):
                    module._weight_quantizer.disable()
                print(f"  - 양자화 비활성화: {name}")


def collect_calibration_stats(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> None:
    """
    Calibration 통계 수집.

    학습 데이터를 사용하여 activation 범위를 측정합니다.

    - Method: MSE (Mean Squared Error) 
    - Num batches: 전체 train 데이터  데이터셋 크기에 맞게 조정)
    - Dataloader: Train dataloader (validation 아님!) 

    Args:
        model: QAT 모델
        data_loader: Calibration용 데이터 로더 (TRAIN 데이터!)
        config: QAT 설정
        device: 디바이스 ('cuda' 또는 'cpu')
    """
    try:
        from pytorch_quantization import calib
    except ImportError:
        raise ImportError("pytorch-quantization이 설치되지 않았습니다.")

    qat_config = config.get('qat', {})
    calib_config = qat_config.get('calibration', {})
    num_batches = calib_config.get('num_batches', 100)  # 기본값 100 (config에서 설정 권장)
    method = calib_config.get('method', 'mse')  # MSE 사용

    print(f"[QAT] Calibration 시작")
    print(f"  - Method: {method}")
    print(f"  - Num batches: {num_batches}")
    print(f"  - Dataloader: Train data (validation 아님!)")

    model.eval()
    model.to(device)

    # Calibration 모드 활성화
    with torch.no_grad():
        _enable_calibration(model)

        for i, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Calibration")):
            if i >= num_batches:
                break

            # 배치 데이터 처리 (ultralytics 형식)
            if isinstance(batch, dict):
                images = batch.get('img', batch.get('image'))
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device).float() / 255.0  # 정규화

            # Forward pass (calibration 통계 수집)
            _ = model(images)

        _disable_calibration(model)

    # Calibration 통계를 기반으로 scale/zero-point 계산
    print(f"[QAT] Calibration 통계 계산 중 (method={method})...")
    _compute_amax(model, method=method)

    print("[QAT] Calibration 완료")


def _enable_calibration(model: nn.Module) -> None:
    """Calibration 모드 활성화"""
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()


def _disable_calibration(model: nn.Module) -> None:
    """Calibration 모드 비활성화"""
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()


def _compute_amax(model: nn.Module, method: str = 'mse') -> None:
    """
    Calibration 통계로부터 amax (activation max) 계산.

    Args:
        model: QAT 모델
        method: Calibration 방법
            - 'mse': Mean Squared Error 
            - 'entropy': KL Divergence 
            - 'max': Absolute Max
            - 'percentile': 99.99 percentile

    """
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization.calib import HistogramCalibrator
    except ImportError:
        return

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                # Calibrator 타입에 따라 다르게 처리
                # HistogramCalibrator만 method 인자 지원
                if isinstance(module._calibrator, HistogramCalibrator):
                    # Medium 기사: method='mse' 사용
                    module.load_calib_amax(method=method)
                else:
                    # MaxCalibrator는 method 인자를 지원하지 않음
                    module.load_calib_amax()

                # Calibrator 메모리 해제
                module._calibrator = None


def disable_quantization(model: nn.Module) -> None:
    """모델의 모든 양자화 비활성화"""
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()


def enable_quantization(model: nn.Module) -> None:
    """모델의 모든 양자화 활성화"""
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()


def get_calibration_dataloader(
    data_yaml: str,
    batch_size: int = 8,
    img_size: int = 640,
    workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Calibration용 DataLoader 생성.

    Args:
        data_yaml: 데이터셋 yaml 파일 경로
        batch_size: 배치 크기
        img_size: 이미지 크기
        workers: DataLoader worker 수

    Returns:
        Calibration용 DataLoader
    """
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import YOLODataset
    import yaml

    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    train_path = data_config.get('train')

    # YOLODataset 생성
    dataset = YOLODataset(
        img_path=train_path,
        imgsz=img_size,
        batch_size=batch_size,
        augment=False,  # Calibration에서는 augmentation 비활성화
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=getattr(dataset, 'collate_fn', None),
    )

    return dataloader


def save_quantizer_state(model: nn.Module) -> Dict[str, Any]:
    """
    TensorQuantizer 상태를 딕셔너리로 저장.

    Checkpoint 저장 시 TensorQuantizer의 중요한 정보(scale, amax 등)를
    함께 저장하여 나중에 복원할 수 있도록 합니다.

    Args:
        model: QAT 모델 (TensorQuantizer 포함)

    Returns:
        {
            'quantizer_count': int,
            'quantizers': {
                'module.path.name': {
                    'num_bits': int,
                    'amax': Tensor,
                    'scale': Tensor,
                    'is_enabled': bool,
                    ...
                }
            }
        }
    """
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return {'quantizer_count': 0, 'quantizers': {}}

    quantizer_state = {
        'quantizer_count': 0,
        'quantizers': {}
    }

    # 모든 TensorQuantizer를 찾아서 상태 저장
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            state = {
                'num_bits': module._num_bits,
                'is_enabled': module._disabled is False,  # enabled = not disabled
            }

            # amax (activation max) 저장
            if hasattr(module, '_amax') and module._amax is not None:
                state['amax'] = module._amax.detach().cpu()

            # scale 저장 (있는 경우)
            if hasattr(module, '_scale') and module._scale is not None:
                state['scale'] = module._scale.detach().cpu()

            # unsigned 여부 저장
            if hasattr(module, '_unsigned'):
                state['unsigned'] = module._unsigned

            # narrow_range 저장
            if hasattr(module, '_narrow_range'):
                state['narrow_range'] = module._narrow_range

            quantizer_state['quantizers'][name] = state
            quantizer_state['quantizer_count'] += 1

    return quantizer_state


def restore_quantizer_state(model: nn.Module, state: Dict[str, Any]) -> None:
    """
    저장된 TensorQuantizer 상태를 모델에 복원.

    Checkpoint에서 로드된 TensorQuantizer 정보를 모델에 다시 적용합니다.

    Args:
        model: QAT 모델 (TensorQuantizer 포함)
        state: save_quantizer_state()로 저장한 상태
    """
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        print("[QAT] pytorch-quantization 미설치, TensorQuantizer 복원 건너뜀")
        return

    if not state or 'quantizers' not in state:
        print("[QAT] ⚠️ 빈 quantizer state, 복원 건너뜀")
        return

    restored_count = 0
    quantizers_in_state = state['quantizers']

    # 모델의 모든 TensorQuantizer를 찾아서 상태 복원
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if name in quantizers_in_state:
                quantizer_state = quantizers_in_state[name]

                # amax 복원
                if 'amax' in quantizer_state:
                    amax = quantizer_state['amax']
                    if isinstance(amax, torch.Tensor):
                        module._amax = amax.to(module._amax.device if module._amax is not None else 'cpu')

                # scale 복원
                if 'scale' in quantizer_state:
                    scale = quantizer_state['scale']
                    if isinstance(scale, torch.Tensor) and hasattr(module, '_scale'):
                        module._scale = scale.to(module._scale.device if module._scale is not None else 'cpu')

                # enabled 상태 복원
                if 'is_enabled' in quantizer_state:
                    if quantizer_state['is_enabled']:
                        module.enable()
                    else:
                        module.disable()

                # unsigned 복원
                if 'unsigned' in quantizer_state and hasattr(module, '_unsigned'):
                    module._unsigned = quantizer_state['unsigned']

                # narrow_range 복원
                if 'narrow_range' in quantizer_state and hasattr(module, '_narrow_range'):
                    module._narrow_range = quantizer_state['narrow_range']

                restored_count += 1

    print(f"[QAT] TensorQuantizer 복원 완료: {restored_count}/{state['quantizer_count']}개")
