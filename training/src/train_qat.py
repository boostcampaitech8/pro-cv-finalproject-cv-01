"""
QAT (Quantization-Aware Training) Trainer

PCBTrainer를 확장하여 QAT 파이프라인을 구현합니다:
1. Calibration: activation 범위 측정
2. QAT Fine-tuning: 양자화 노이즈 포함 학습
3. ONNX Export: Q/DQ 노드 포함
"""

import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from ultralytics import YOLO
from src.train import PCBTrainer


class QATTrainer(PCBTrainer):
    """
    QAT (Quantization-Aware Training) 트레이너.

    PCBTrainer를 상속받아 calibration과 QAT fine-tuning을 추가합니다.
    """

    def __init__(self, model: YOLO, config: Dict[str, Any]):
        """
        Args:
            model: YOLO 모델 (pre-trained)
            config: QAT 설정
        """
        super().__init__(model, config)
        self.qat_enabled = self._check_qat_available()

    def _check_qat_available(self) -> bool:
        """pytorch-quantization 사용 가능 여부 확인"""
        try:
            from pytorch_quantization import quant_modules
            return True
        except ImportError:
            print("[QAT] 경고: pytorch-quantization이 설치되지 않음")
            print("  QAT 없이 일반 fine-tuning으로 진행합니다.")
            return False

    def calibrate(self, data_yaml: str) -> None:
        """
        Calibration 수행.

        학습 데이터의 일부를 사용하여 activation 범위를 측정합니다.

        Args:
            data_yaml: 데이터셋 yaml 파일 경로
        """
        if not self.qat_enabled:
            print("[QAT] Calibration 건너뜀 (QAT 비활성화)")
            return

        print(f"\n{'='*20} QAT Calibration {'='*20}")

        try:
            from src.quantization import collect_calibration_stats
            from src.quantization.qat_utils import get_calibration_dataloader

            qat_config = self.config.get('qat', {})
            calib_config = qat_config.get('calibration', {})
            num_batches = calib_config.get('num_batches', 100)

            # Calibration용 데이터로더 생성
            batch_size = self.config.get('batch_size', 8)
            img_size = self.config.get('img_size', 640)
            workers = self.config.get('workers', 4)

            print(f"[QAT] Calibration 데이터로더 생성...")
            print(f"  - Batch size: {batch_size}")
            print(f"  - Num batches: {num_batches}")

            # ultralytics의 내부 데이터로더 사용
            dataloader = self._get_ultralytics_dataloader(data_yaml, batch_size, img_size)

            # Calibration 수행
            device = self.config.get('device', '0')
            if device != 'cpu':
                device = f'cuda:{device}' if not device.startswith('cuda') else device
            else:
                device = 'cpu'

            collect_calibration_stats(
                model=self.model.model,  # YOLO 내부 PyTorch 모델
                data_loader=dataloader,
                config=self.config,
                device=device
            )

            print(f"{'='*50}\n")

        except ImportError as e:
            print(f"[QAT] Calibration 실패: {e}")
        except Exception as e:
            print(f"[QAT] Calibration 오류: {e}")
            import traceback
            traceback.print_exc()

    def _get_ultralytics_dataloader(
        self,
        data_yaml: str,
        batch_size: int,
        img_size: int
    ):
        """
        Ultralytics 스타일 데이터로더 생성.

        calibration에 사용할 학습 데이터 로더를 생성합니다.
        """
        import yaml
        from ultralytics.data import build_dataloader, build_yolo_dataset

        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)

        # 학습 이미지 경로
        train_path = data.get('train')
        if not os.path.isabs(train_path):
            data_root = os.path.dirname(data_yaml)
            train_path = os.path.abspath(os.path.join(data_root, train_path))

        # 간단한 PyTorch DataLoader 생성
        from torch.utils.data import DataLoader
        from ultralytics.data.dataset import YOLODataset

        dataset = YOLODataset(
            img_path=train_path,
            imgsz=img_size,
            batch_size=batch_size,
            augment=False,
            rect=False,
            stride=32,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(self.config.get('workers', 4), 4),
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        return dataloader

    def train(self, data_yaml: str) -> Optional[Path]:
        """
        QAT Fine-tuning 수행.

        calibration 후 양자화 노이즈를 포함한 학습을 수행합니다.

        Args:
            data_yaml: 데이터셋 yaml 파일 경로

        Returns:
            학습 결과 저장 디렉토리
        """
        print(f"\n{'='*20} QAT Fine-tuning {'='*20}")

        # 양자화 활성화
        if self.qat_enabled:
            try:
                from src.quantization import enable_quantization
                enable_quantization(self.model.model)
                print("[QAT] 양자화 활성화됨")
            except Exception as e:
                print(f"[QAT] 양자화 활성화 실패: {e}")

        # 부모 클래스의 train 호출 (PCBTrainer.train)
        save_dir = super().train(data_yaml)

        # ONNX Export
        if save_dir:
            self._export_onnx(save_dir, data_yaml)

        return save_dir

    def _export_onnx(self, save_dir: Path, data_yaml: str) -> None:
        """
        QAT 학습 후 ONNX export.

        Args:
            save_dir: 학습 결과 저장 디렉토리
            data_yaml: 데이터셋 yaml 경로
        """
        print(f"\n{'='*20} ONNX Export {'='*20}")

        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        if not os.path.exists(best_model_path):
            print(f"[QAT] best.pt를 찾을 수 없음: {best_model_path}")
            return

        try:
            # best.pt 로드
            best_model = YOLO(best_model_path)

            # ONNX export 경로
            onnx_path = os.path.join(save_dir, "weights", "best_qat.onnx")

            if self.qat_enabled:
                # QAT ONNX export (Q/DQ 노드 포함)
                from src.quantization import export_qat_to_onnx
                export_qat_to_onnx(
                    model=best_model.model,
                    output_path=onnx_path,
                    config=self.config,
                    img_size=self.config.get('img_size', 640),
                )
            else:
                # 일반 ONNX export
                best_model.export(
                    format='onnx',
                    imgsz=self.config.get('img_size', 640),
                    simplify=True,
                )
                # 파일명 변경
                default_onnx = best_model_path.replace('.pt', '.onnx')
                if os.path.exists(default_onnx):
                    os.rename(default_onnx, onnx_path)

            if os.path.exists(onnx_path):
                print(f"[QAT] ONNX 저장 완료: {onnx_path}")

        except Exception as e:
            print(f"[QAT] ONNX export 실패: {e}")
            import traceback
            traceback.print_exc()

    def run_full_pipeline(self, data_yaml: str) -> Optional[Path]:
        """
        전체 QAT 파이프라인 실행.

        1. Calibration
        2. QAT Fine-tuning
        3. ONNX Export
        4. Validation

        Args:
            data_yaml: 데이터셋 yaml 파일 경로

        Returns:
            학습 결과 저장 디렉토리
        """
        print(f"\n{'='*50}")
        print(f"QAT Pipeline Start")
        print(f"{'='*50}\n")

        start_time = time.time()

        # 1. Calibration
        self.calibrate(data_yaml)

        # 2. QAT Fine-tuning + ONNX Export
        save_dir = self.train(data_yaml)

        # 3. 최종 결과 요약
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"QAT Pipeline Complete")
        print(f"  - Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        if save_dir:
            print(f"  - Results: {save_dir}")
            print(f"  - ONNX: {save_dir}/weights/best_qat.onnx")
        print(f"{'='*50}\n")

        return save_dir
