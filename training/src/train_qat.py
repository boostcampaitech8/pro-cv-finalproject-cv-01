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
            data=data,  # data.yaml 설정 (필수)
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
            num_workers=self.config.get('workers', 8),
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
        print(f"\n{'='*20} QAT Training {'='*20}")

        # 양자화 활성화
        if self.qat_enabled:
            try:
                from src.quantization import enable_quantization
                enable_quantization(self.model.model)
                print("[QAT] 양자화 활성화됨")
            except Exception as e:
                print(f"[QAT] 양자화 활성화 실패: {e}")

        # QAT Callback 등록 (best 모델 저장 시 Raw checkpoint 저장)
        if self.qat_enabled:
            self._register_qat_callback()

        # 부모 클래스의 train 호출 (PCBTrainer.train)
        save_dir = super().train(data_yaml)

        # 학습 완료 후 Raw checkpoint에서 ONNX export
        if save_dir and self.qat_enabled:
            self._export_onnx_from_raw_checkpoint(save_dir)

        return save_dir

    def _register_qat_callback(self) -> None:
        """
        QAT Raw Checkpoint 저장 callback 등록.

        best 모델이 업데이트될 때마다 Raw PyTorch checkpoint를 저장합니다.
        이렇게 하면 TensorQuantizer 정보가 온전히 보존됩니다.
        """
        def on_fit_epoch_end(trainer):
            """Epoch 종료 시 best 모델이면 Raw checkpoint 저장"""
            try:
                # best 모델이 업데이트되었는지 확인
                # trainer.best_fitness가 현재 fitness와 같으면 best 업데이트됨
                if not hasattr(trainer, 'best_fitness'):
                    return

                if not hasattr(trainer, 'fitness'):
                    return

                # Best 모델 업데이트 확인
                is_best = (trainer.best_fitness == trainer.fitness)

                if is_best:
                    epoch = trainer.epoch if hasattr(trainer, 'epoch') else 'unknown'
                    print(f"\n{'='*60}")
                    print(f"[QAT] Best 모델 업데이트 감지! (Epoch {epoch})")
                    print(f"{'='*60}")

                    # TensorQuantizer 확인
                    from pytorch_quantization import nn as quant_nn
                    quantizer_count = sum(1 for m in self.model.model.modules()
                                         if isinstance(m, quant_nn.TensorQuantizer))
                    print(f"[QAT] TensorQuantizer 개수: {quantizer_count}")

                    if quantizer_count == 0:
                        print(f"[QAT] ⚠️ 경고: TensorQuantizer가 없습니다!")
                        print(f"[QAT] 모델이 이미 변환되었을 수 있습니다.")

                    # Raw checkpoint 저장 경로
                    save_dir = trainer.save_dir if hasattr(trainer, 'save_dir') else self.config.get('project', './runs')
                    weights_dir = os.path.join(save_dir, "weights")

                    # 디렉토리 생성
                    os.makedirs(weights_dir, exist_ok=True)

                    raw_path = os.path.join(weights_dir, "best_qat_raw.pt")

                    print(f"[QAT] Raw checkpoint 저장 중: {raw_path}")

                    # 전체 모델 객체 저장 (TensorQuantizer 보존)
                    torch.save({
                        'model': self.model.model,  # 전체 모델 객체
                        'epoch': epoch,
                        'fitness': trainer.fitness if hasattr(trainer, 'fitness') else None,
                        'best_fitness': trainer.best_fitness if hasattr(trainer, 'best_fitness') else None,
                    }, raw_path)

                    # 저장 확인
                    if os.path.exists(raw_path):
                        file_size = os.path.getsize(raw_path) / 1024 / 1024
                        print(f"[QAT] ✅ Raw checkpoint 저장 완료!")
                        print(f"  - 경로: {raw_path}")
                        print(f"  - 크기: {file_size:.1f} MB")
                        print(f"  - TensorQuantizer: {quantizer_count}개")
                    else:
                        print(f"[QAT] ❌ 저장 실패: 파일이 생성되지 않음")

                    print(f"{'='*60}\n")

            except Exception as e:
                print(f"[QAT] ❌ Raw checkpoint 저장 실패: {e}")
                import traceback
                traceback.print_exc()

        # Callback 등록
        try:
            self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            print("[QAT] ✅ Raw checkpoint 저장 callback 등록 완료")
        except Exception as e:
            print(f"[QAT] ⚠️ Callback 등록 실패: {e}")
            print(f"[QAT] 학습은 계속되지만 Raw checkpoint가 저장되지 않을 수 있습니다.")

    def _export_onnx_from_raw_checkpoint(self, save_dir: Path) -> None:
        """
        학습 완료 후 Raw checkpoint에서 ONNX export.

        Args:
            save_dir: 학습 결과 저장 디렉토리
        """
        print(f"\n{'='*60}")
        print(f"[QAT] Raw Checkpoint → ONNX Export")
        print(f"{'='*60}")

        raw_path = os.path.join(save_dir, "weights", "best_qat_raw.pt")
        onnx_path = os.path.join(save_dir, "weights", "best_qat.onnx")

        # Raw checkpoint 존재 확인
        if not os.path.exists(raw_path):
            print(f"[QAT] ⚠️ Raw checkpoint를 찾을 수 없음: {raw_path}")
            print(f"[QAT] Callback이 실행되지 않았거나 저장에 실패했습니다.")
            return

        print(f"[QAT] Raw checkpoint 로드 중: {raw_path}")

        try:
            # Raw checkpoint 로드
            checkpoint = torch.load(raw_path, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model = checkpoint['model']
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"[QAT] Checkpoint 로드 완료 (Epoch {epoch})")
            else:
                model = checkpoint
                print(f"[QAT] Checkpoint 로드 완료")

            # TensorQuantizer 확인
            from pytorch_quantization import nn as quant_nn
            quantizer_count = sum(1 for m in model.modules()
                                 if isinstance(m, quant_nn.TensorQuantizer))
            print(f"[QAT] TensorQuantizer 개수: {quantizer_count}")

            if quantizer_count == 0:
                print(f"\n[QAT] ❌ TensorQuantizer가 없습니다!")
                print(f"[QAT] Raw checkpoint에도 QAT 정보가 없습니다.")
                print(f"[QAT] 학습 시 QAT가 제대로 적용되지 않았을 가능성이 있습니다.")
                return

            print(f"[QAT] ✅ TensorQuantizer 보존됨!")
            print(f"\n[QAT] ONNX Export 시작...")

            # ONNX export
            from src.quantization import export_qat_to_onnx
            export_qat_to_onnx(
                model=model,
                output_path=onnx_path,
                config=self.config,
                img_size=self.config.get('img_size', 640),
            )

            if os.path.exists(onnx_path):
                file_size = os.path.getsize(onnx_path) / 1024 / 1024
                print(f"\n[QAT] ✅ ONNX export 성공!")
                print(f"  - 경로: {onnx_path}")
                print(f"  - 크기: {file_size:.1f} MB")

                # Q/DQ 노드 확인
                try:
                    import onnx
                    onnx_model = onnx.load(onnx_path)
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
                        print(f"[QAT] ONNX export 과정에서 Q/DQ 노드가 손실되었을 수 있습니다.")
                except Exception as e:
                    print(f"[QAT] Q/DQ 노드 확인 실패: {e}")

        except Exception as e:
            print(f"\n[QAT] ❌ ONNX export 실패: {e}")
            import traceback
            traceback.print_exc()

        print(f"{'='*60}\n")

    def _export_onnx_from_memory(self, save_dir: Path, data_yaml: str) -> None:
        """
        메모리 상의 QAT 모델을 직접 ONNX export.

        학습 직후 메모리에 있는 self.model.model을 사용하여 QAT 정보를 보존합니다.

        Args:
            save_dir: 학습 결과 저장 디렉토리
            data_yaml: 데이터셋 yaml 경로
        """
        print(f"\n{'='*20} ONNX Export (from memory) {'='*20}")

        onnx_path = os.path.join(save_dir, "weights", "best_qat.onnx")

        if not self.qat_enabled:
            print("[QAT] QAT 비활성화됨. 일반 ONNX export 건너뜀.")
            return

        try:
            # 메모리 상의 모델 사용 (QAT 정보 보존)
            from src.quantization import export_qat_to_onnx

            print("[QAT] 메모리 상의 QAT 모델을 ONNX로 export 중...")
            print(f"[QAT] TensorQuantizer 확인...")

            # TensorQuantizer 개수 확인
            from pytorch_quantization import nn as quant_nn
            quantizer_count = sum(1 for m in self.model.model.modules()
                                 if isinstance(m, quant_nn.TensorQuantizer))
            print(f"[QAT] TensorQuantizer 개수: {quantizer_count}")

            if quantizer_count == 0:
                print("[QAT] ⚠️ 경고: TensorQuantizer가 없습니다. QAT가 적용되지 않았을 수 있습니다.")

            export_qat_to_onnx(
                model=self.model.model,  # 메모리 상의 모델 직접 사용
                output_path=onnx_path,
                config=self.config,
                img_size=self.config.get('img_size', 640),
            )

            if os.path.exists(onnx_path):
                print(f"[QAT] ✅ ONNX 저장 완료: {onnx_path}")

        except Exception as e:
            print(f"[QAT] ❌ ONNX export 실패: {e}")
            import traceback
            traceback.print_exc()

    def _export_onnx(self, save_dir: Path, data_yaml: str) -> None:
        """
        QAT 학습 후 ONNX export (레거시 메서드, 더 이상 사용 안 함).

        Args:
            save_dir: 학습 결과 저장 디렉토리
            data_yaml: 데이터셋 yaml 경로
        """
        # 이 메서드는 더 이상 사용하지 않습니다.
        # _export_onnx_from_memory를 대신 사용합니다.
        pass

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
