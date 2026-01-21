"""
QAT용 커스텀 DetectionTrainer

- Ultralytics DetectionTrainer를 직접 상속
- _setup_train() 오버라이드로 QAT 전용 설정 적용
- warmup_epochs = 0, amp = False, CosineAnnealingLR 스케줄러
"""

import torch
import torch.optim as optim
from pathlib import Path
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG_DICT


class QATDetectionTrainer(DetectionTrainer):
    """
    QAT용 커스텀 DetectionTrainer.

    _setup_train()을 오버라이드하여 QAT 전용 설정을 적용합니다:
    - warmup_epochs = 0 (QAT는 warmup 불필요)
    - amp = False (AMP는 Q/DQ 노드와 충돌)
    - CosineAnnealingLR 스케줄러
    - 낮은 learning rate (원래 lr의 1/100)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        Args:
            cfg: 기본 설정 딕셔너리 (None이면 DEFAULT_CFG_DICT 사용)
            overrides: 설정 오버라이드 (train_args 딕셔너리)
            _callbacks: Callback 함수들
        """
        # cfg가 None이면 기본 설정 사용
        if cfg is None:
            cfg = DEFAULT_CFG_DICT.copy()

        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        # QAT 전용 하이퍼파라미터
        self.qat_epochs = 20
        self.qat_lr = 0.0001  # 원래 lr의 1/100 정도

        print("[QAT Trainer] 초기화 완료")
        print(f"  - QAT epochs: {self.qat_epochs}")
        print(f"  - QAT lr: {self.qat_lr}")

    def _setup_train(self, world_size=1):
        """
        QAT 전용 학습 설정.

        부모 클래스의 _setup_train()을 호출한 후,
        QAT에 필요한 설정을 오버라이드합니다.

        Args:
            world_size: 분산 학습 world size (기본값: 1)
        """
        # 부모 클래스의 _setup_train 호출 (기본 설정)
        super()._setup_train()

        print("\n" + "="*60)
        print("[QAT Trainer] _setup_train() 오버라이드 시작")
        print("="*60)

        # 1. Warmup 비활성화 (QAT는 fine-tuning이므로 warmup 불필요)
        self.args.warmup_epochs = 0
        print(f"[QAT] warmup_epochs: {self.args.warmup_epochs} (비활성화)")

        # 2. AMP 비활성화 (중요! Q/DQ 노드 추가 후 half precision 오류 방지)
        self.args.amp = False
        self.amp = False
        print(f"[QAT] amp: {self.amp} (비활성화 - 필수!)")

        # 3. QAT 하이퍼파라미터 설정
        self.epochs = self.qat_epochs
        self.args.lr0 = self.qat_lr
        print(f"[QAT] epochs: {self.epochs}")
        print(f"[QAT] lr0: {self.args.lr0}")

        # 4. CosineAnnealingLR 스케줄러로 교체
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs * 1.0,
            eta_min=self.args.lr0 * 0.01  # 최소 lr = lr0 * 0.01
        )
        print(f"[QAT] scheduler: CosineAnnealingLR")
        print(f"  - T_max: {self.epochs}")
        print(f"  - eta_min: {self.args.lr0 * 0.01}")

        # 5. TensorQuantizer 개수 확인 (디버깅)
        try:
            from pytorch_quantization import nn as quant_nn
            quantizer_count = sum(1 for m in self.model.modules()
                                 if isinstance(m, quant_nn.TensorQuantizer))
            print(f"\n[QAT] TensorQuantizer 개수: {quantizer_count}")

            if quantizer_count == 0:
                print(f"[QAT] ⚠️ 경고: TensorQuantizer가 없습니다!")
                print(f"  모델이 QAT용으로 준비되지 않았을 수 있습니다.")
            else:
                print(f"[QAT] ✅ TensorQuantizer 확인됨!")
        except ImportError:
            print(f"[QAT] pytorch-quantization 미설치 (QAT 비활성화)")

        print("="*60 + "\n")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        모델 로드 (QAT checkpoint 우선 처리, strict=False 사용).

        **핵심 변경사항**:
        1. QAT checkpoint 감지 시, 빈 모델 생성 → QuantConv2d 교체 → state_dict 로드 순서
        2. strict=False 사용으로 키 불일치 허용 (TensorQuantizer 키 누락 무시)
        3. FP32로 저장된 checkpoint를 그대로 로드

        Args:
            cfg: 모델 config
            weights: 가중치 경로
            verbose: 로그 출력 여부

        Returns:
            모델
        """
        # QAT checkpoint 확인
        if weights and Path(weights).exists():
            try:
                checkpoint = torch.load(weights, map_location='cpu', weights_only=False)

                if 'quantizer_state' in checkpoint:
                    print(f"\n[QAT] QAT Checkpoint 감지: {Path(weights).name}")
                    print(f"[QAT] 참조 리포지토리 방식으로 로드 중...")

                    # 1단계: 빈 모델 생성 (weights 없이)
                    print(f"[QAT] [1/3] 빈 모델 생성...")
                    model = super().get_model(cfg=cfg, weights=None, verbose=verbose)

                    # 2단계: Conv2d → QuantConv2d 교체
                    print(f"[QAT] [2/3] Conv2d → QuantConv2d 교체...")
                    quantizer_state = checkpoint['quantizer_state']

                    # num_bits 추출
                    num_bits = 8
                    if 'quantizers' in quantizer_state and quantizer_state['quantizers']:
                        first_quantizer = next(iter(quantizer_state['quantizers'].values()))
                        num_bits = first_quantizer.get('num_bits', 8)

                    # QAT config 생성
                    qat_config = {
                        'qat': {
                            'quantization': {
                                'num_bits': num_bits,
                                'weight_per_channel': True,
                            },
                            'calibration': {
                                'method': 'histogram'
                            }
                        }
                    }

                    from src.quantization.qat_utils import replace_conv_with_quantconv
                    model = replace_conv_with_quantconv(model, qat_config)

                    # 3단계: state_dict 로드 (strict=False로 키 불일치 허용)
                    print(f"[QAT] [3/3] State dict 로드 (strict=False)...")

                    # checkpoint['model']이 QuantConv2d 상태를 포함하고 있으므로 직접 로드
                    missing_keys, unexpected_keys = model.load_state_dict(
                        checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model'],
                        strict=False
                    )

                    print(f"[QAT] State dict 로드 완료")
                    if missing_keys:
                        print(f"  - Missing keys: {len(missing_keys)} (정상, TensorQuantizer 내부 키)")
                    if unexpected_keys:
                        print(f"  - Unexpected keys: {len(unexpected_keys)}")

                    # TensorQuantizer 개수 확인
                    from pytorch_quantization import nn as quant_nn
                    quantizer_count = sum(1 for m in model.modules()
                                         if isinstance(m, quant_nn.TensorQuantizer))
                    print(f"[QAT] ✅ QAT 모델 복원 완료: {quantizer_count}개 TensorQuantizer\n")

                    return model

                else:
                    print(f"[QAT] 일반 checkpoint (quantizer_state 없음)")

            except Exception as e:
                print(f"[QAT] ⚠️ QAT 모델 복원 실패: {e}")
                import traceback
                traceback.print_exc()
                print(f"[QAT] 일반 모델로 폴백합니다...")

        # 일반 checkpoint 또는 QAT 로드 실패 시
        return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

    def save_model(self):
        """
        QAT 모델 저장 (.half() 제거, FP32 유지).

        Ultralytics BaseTrainer.save_model()을 완전히 오버라이드하여
        .half() 호출을 방지합니다. QuantConv2d + TensorQuantizer를
        FP32 그대로 저장하여 precision 손실을 막습니다.
        """
        from copy import deepcopy
        from ultralytics.utils.torch_utils import unwrap_model
        from ultralytics import __version__
        from datetime import datetime
        import pandas as pd

        # 메트릭 수집 (부모 클래스 로직 복사)
        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}

        # TensorQuantizer 상태 저장
        from src.quantization.qat_utils import save_quantizer_state
        quantizer_state = save_quantizer_state(self.model)

        # Checkpoint 딕셔너리 생성 (.half() 제거!)
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(unwrap_model(self.model)),  # ← .half() 제거!
            'ema': deepcopy(unwrap_model(self.ema.ema)) if self.ema else None,  # ← .half() 제거!
            'updates': self.ema.updates if self.ema else None,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'version': __version__,
            'quantizer_state': quantizer_state,  # ← TensorQuantizer 정보 추가
        }

        # 저장 (부모 클래스 로직 복사)
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')

        print(f"[QAT] ✅ Checkpoint 저장 완료 (FP32, QuantConv2d 유지)")
        print(f"  - TensorQuantizer: {quantizer_state['quantizer_count']}개")
        print(f"  - Best: {self.best}")
        print(f"  - Last: {self.last}")

    def optimizer_step(self):
        """
        Optimizer step 수행.

        부모 클래스의 optimizer_step()을 그대로 사용하되,
        gradient clipping 등의 추가 로직을 여기에 넣을 수 있습니다.
        """
        # 부모 클래스의 optimizer step
        super().optimizer_step()

        # CosineAnnealingLR 스케줄러는 epoch 단위가 아니라
        # batch 단위로 업데이트할 수도 있음 (선택사항)
        # self.scheduler.step()  # 이미 부모 클래스에서 호출됨

    def _do_train(self):
        """
        실제 학습 루프 수행.

        부모 클래스의 _do_train()을 그대로 사용합니다.
        QAT 설정은 _setup_train()에서 이미 적용되었습니다.
        """
        return super()._do_train()
