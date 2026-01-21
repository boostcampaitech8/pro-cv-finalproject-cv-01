"""
QAT 학습 스크립트

워크플로우:
1. 일반 모델 로드 (pre-trained best.pt)
2. Q/DQ 노드 추가 (Conv2d → QuantConv2d)
3. Calibration (MSE, train loader)
4. 커스텀 트레이너로 QAT fine-tuning (20 epochs, warmup=0, amp=False, CosineAnnealingLR)
5. ONNX export (Q/DQ 노드 포함)
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add training directory to Python path
# training/src/train_qat_custom.py에서 실행되므로 부모의 부모 디렉토리를 추가
training_dir = Path(__file__).parent.parent
sys.path.insert(0, str(training_dir))

from ultralytics import YOLO


def run_qat_training(config: Dict[str, Any], data_yaml: str) -> Optional[Path]:
    """
    QAT 학습 수행.

    Args:
        config: QAT 설정 (config_qat.yaml)
        data_yaml: 데이터셋 yaml 경로

    Returns:
        학습 결과 저장 디렉토리
    """
    print("\n" + "="*80)
    print("QAT Training")
    print("="*80 + "\n")

    # ========================================================================
    # 1. Pre-trained 모델 로드
    # ========================================================================
    print("\n[1/5] Pre-trained 모델 로드...")
    print("-"*80)

    qat_config = config.get('qat', {})
    pretrained_path = qat_config.get('pretrained_path', '')

    # 절대 경로로 변환
    if pretrained_path and not os.path.isabs(pretrained_path):
        base_dir = Path(__file__).parent.parent.parent
        pretrained_path = str(base_dir / pretrained_path)

    # 존재 확인
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pre-trained 모델을 찾을 수 없습니다: {pretrained_path}\n"
            "먼저 일반 학습을 수행하세요: uv run python run_exp.py --config config.yaml"
        )

    print(f"[QAT] Pre-trained 모델: {pretrained_path}")
    model = YOLO(pretrained_path)
    print(f"[QAT] ✅ 모델 로드 완료")
    print(f"  - Classes: {len(model.names)}")
    print(f"  - Names: {list(model.names.values())}")

    # ========================================================================
    # 2. Q/DQ 노드 추가 (Conv2d → QuantConv2d)
    # ========================================================================
    print("\n[2/5] Q/DQ 노드 추가 (Conv2d → QuantConv2d)...")
    print("-"*80)

    try:
        from src.quantization import replace_conv_with_quantconv, disable_detect_head_quantization
        from pytorch_quantization import nn as quant_nn

        # Conv2d를 QuantConv2d로 수동 교체
        print("[QAT] replace_conv_with_quantconv() 호출 중...")
        replace_conv_with_quantconv(model.model, config)

        # Detect Head 양자화 비활성화 (정확도 확보)
        print("[QAT] Detect Head 양자화 비활성화 중...")
        disable_detect_head_quantization(model.model)

        # 교체 확인
        quantizer_count = sum(1 for m in model.model.modules()
                             if isinstance(m, quant_nn.TensorQuantizer))
        print(f"\n[QAT] ✅ 양자화 교체 완료!")
        print(f"  - TensorQuantizer 개수: {quantizer_count}")

        if quantizer_count == 0:
            raise RuntimeError("QAT 모듈 교체 실패: TensorQuantizer가 없음")

    except ImportError as e:
        print(f"\n[QAT] ❌ ImportError: {e}")
        print(f"  pytorch-quantization을 설치하세요:")
        print(f"  pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com")
        raise
    except Exception as e:
        print(f"\n[QAT] ❌ 양자화 변환 오류: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ========================================================================
    # 3. Calibration (MSE, 1024 batch, train loader)
    # ========================================================================
    print("\n[3/5] Calibration (MSE, Train loader)...")
    print("-"*80)

    try:
        from src.quantization import collect_calibration_stats

        # Train dataloader 생성
        print("[QAT] Train dataloader 생성 중...")
        train_loader = _get_train_dataloader(data_yaml, config)

        # Calibration 수행
        device = config.get('device', '0')
        if device != 'cpu':
            device = f'cuda:{device}' if not device.startswith('cuda') else device
        else:
            device = 'cpu'

        collect_calibration_stats(
            model=model.model,  # YOLO 내부 PyTorch 모델
            data_loader=train_loader,
            config=config,
            device=device
        )

        print(f"[QAT] ✅ Calibration 완료")

    except Exception as e:
        print(f"\n[QAT] ❌ Calibration 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ========================================================================
    # 4. 커스텀 트레이너로 QAT Fine-tuning
    # ========================================================================
    print("\n[4/5] QAT Fine-tuning (커스텀 트레이너)...")
    print("-"*80)

    try:
        from src.models.qat_detection_trainer import QATDetectionTrainer

        # 양자화 활성화
        from src.quantization import enable_quantization
        enable_quantization(model.model)
        print("[QAT] 양자화 활성화 완료")

        # 커스텀 트레이너 생성
        print("[QAT] 커스텀 트레이너 생성 중...")

        # Ultralytics 형식 args 준비
        train_args = {
            'data': data_yaml,
            'epochs': 20,
            'batch': config.get('batch_size', 8),
            'imgsz': config.get('img_size', 640),
            'device': device,
            'workers': config.get('workers', 8),
            'project': config.get('project', 'runs/qat'),
            'name': config.get('exp_name', 'qat_yolov8s'),
            'exist_ok': True,
            'pretrained': False,  # 이미 로드된 모델 사용
            'optimizer': config.get('optimizer', 'AdamW'),
            'lr0': 0.0001,  # 원래 lr의 1/100
            'lrf': 0.01,
            'momentum': config.get('momentum', 0.937),
            'weight_decay': config.get('weight_decay', 0.0005),
            'warmup_epochs': 0,  # 커스텀 트레이너에서 설정됨
            'amp': False,  # 커스텀 트레이너에서 설정됨
            'patience': config.get('patience', 30),
            'save': config.get('save', True),
            'save_period': config.get('save_period', -1),
            'box': config.get('box', 7.5),
            'cls': config.get('cls', 0.5),
            'dfl': config.get('dfl', 1.5),
            'hsv_h': config.get('hsv_h', 0.015),
            'hsv_s': config.get('hsv_s', 0.7),
            'hsv_v': config.get('hsv_v', 0.4),
            'degrees': config.get('degrees', 0.0),
            'translate': config.get('translate', 0.1),
            'scale': config.get('scale', 0.5),
            'shear': config.get('shear', 0.0),
            'perspective': config.get('perspective', 0.0),
            'flipud': config.get('flipud', 0.0),
            'fliplr': config.get('fliplr', 0.5),
            'mosaic': config.get('mosaic', 1.0),
            'mixup': config.get('mixup', 0.0),
            'copy_paste': config.get('copy_paste', 0.0),
        }

        # 커스텀 트레이너로 학습
        trainer = QATDetectionTrainer(overrides=train_args)
        trainer.model = model.model  # QAT 모델 할당
        trainer.train()

        save_dir = Path(trainer.save_dir)
        print(f"\n[QAT] ✅ QAT Fine-tuning 완료")
        print(f"  - Save dir: {save_dir}")

    except Exception as e:
        print(f"\n[QAT] ❌ QAT Fine-tuning 실패: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ========================================================================
    # 5. ONNX Export (Q/DQ 노드 포함)
    # ========================================================================
    print("\n[5/5] ONNX Export (Q/DQ 노드 포함)...")
    print("-"*80)

    try:
        from src.quantization import export_qat_to_onnx

        # Best checkpoint 경로
        best_pt = save_dir / "weights" / "best.pt"
        onnx_path = save_dir / "weights" / "best_qat.onnx"

        if not best_pt.exists():
            print(f"[QAT] ⚠️ Best checkpoint를 찾을 수 없음: {best_pt}")
            print("[QAT] 학습이 완료되지 않았거나 저장에 실패했습니다.")
            return save_dir

        # ONNX export
        print(f"[QAT] Best checkpoint: {best_pt}")
        print(f"[QAT] ONNX export 중...")

        # Checkpoint 로드
        checkpoint = torch.load(best_pt, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            export_model = checkpoint['model']
        else:
            export_model = checkpoint

        # TensorQuantizer 확인
        quantizer_count = sum(1 for m in export_model.modules()
                             if isinstance(m, quant_nn.TensorQuantizer))
        print(f"[QAT] TensorQuantizer 개수: {quantizer_count}")

        if quantizer_count == 0:
            print(f"\n[QAT] ❌ TensorQuantizer가 없습니다!")
            print(f"  Best checkpoint에 QAT 정보가 없을 수 있습니다.")
            # 메모리 상의 모델 사용
            print(f"[QAT] 메모리 상의 모델로 export 시도...")
            export_model = model.model

        # ONNX export
        export_qat_to_onnx(
            model=export_model,
            output_path=str(onnx_path),
            config=config,
            img_size=config.get('img_size', 640),
        )

        if onnx_path.exists():
            file_size = onnx_path.stat().st_size / 1024 / 1024
            print(f"\n[QAT] ✅ ONNX export 성공!")
            print(f"  - 경로: {onnx_path}")
            print(f"  - 크기: {file_size:.1f} MB")

            # Q/DQ 노드 확인
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                qdq_count = {}
                for node in onnx_model.graph.node:
                    if node.op_type in ['QuantizeLinear', 'DequantizeLinear']:
                        qdq_count[node.op_type] = qdq_count.get(node.op_type, 0) + 1

                if qdq_count:
                    print(f"\n[QAT] 🎉 Q/DQ 노드 발견!")
                    for op, count in qdq_count.items():
                        print(f"  - {op}: {count}개")
                    print(f"\n[QAT] ✅ 엣지에서 calibration 불필요!")
                else:
                    print(f"\n[QAT] ⚠️ Q/DQ 노드 없음")
                    print(f"  ONNX export 과정에서 Q/DQ 노드가 손실되었을 수 있습니다.")
            except Exception as e:
                print(f"[QAT] Q/DQ 노드 확인 실패: {e}")

    except Exception as e:
        print(f"\n[QAT] ❌ ONNX export 실패: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # 완료
    # ========================================================================
    print("\n" + "="*80)
    print("QAT Training 완료!")
    print("="*80)
    print(f"  - Save dir: {save_dir}")
    print(f"  - Best checkpoint: {save_dir}/weights/best.pt")
    print(f"  - ONNX: {save_dir}/weights/best_qat.onnx")
    print("="*80 + "\n")

    return save_dir


def _get_train_dataloader(data_yaml: str, config: Dict[str, Any]):
    """
    Train dataloader 생성 (Calibration용).

    Args:
        data_yaml: 데이터셋 yaml 경로
        config: 설정

    Returns:
        Train dataloader
    """
    import yaml
    from torch.utils.data import DataLoader
    from ultralytics.data.dataset import YOLODataset

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # 학습 이미지 경로
    train_path = data.get('train')
    if not os.path.isabs(train_path):
        data_root = os.path.dirname(data_yaml)
        train_path = os.path.abspath(os.path.join(data_root, train_path))

    # YOLODataset 생성
    dataset = YOLODataset(
        img_path=train_path,
        data=data,
        imgsz=config.get('img_size', 640),
        batch_size=config.get('batch_size', 8),
        augment=False,  # Calibration에서는 augmentation 비활성화
        rect=False,
        stride=32,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('workers', 8),
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    print(f"[QAT] Train dataloader 생성 완료")
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - Batch size: {config.get('batch_size', 8)}")
    print(f"  - Total batches: {len(dataloader)}")

    return dataloader


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='QAT Training')
    parser.add_argument('--config', type=str, default='config_qat.yaml', help='Config file path')
    parser.add_argument('--data', type=str, default=None, help='Data yaml path (optional)')
    args = parser.parse_args()

    # Config 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Data yaml 경로
    if args.data:
        data_yaml = args.data
    else:
        # Config에서 data_path 사용
        data_path = config.get('data_path', './PCB_DATASET')
        data_yaml = os.path.join(data_path, 'data.yaml')

    # QAT 학습 실행
    save_dir = run_qat_training(config, data_yaml)

    if save_dir:
        print(f"\n✅ QAT 학습 완료: {save_dir}")
    else:
        print(f"\n❌ QAT 학습 실패")
        sys.exit(1)
