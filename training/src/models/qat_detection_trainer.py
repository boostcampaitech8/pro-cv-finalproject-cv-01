"""
QAT용 커스텀 DetectionTrainer

- Ultralytics DetectionTrainer를 직접 상속
- _setup_train() 오버라이드로 QAT 전용 설정 적용
- warmup_epochs = 0, amp = False, CosineAnnealingLR 스케줄러
"""

import torch.optim as optim
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

    def _setup_train(self, world_size):
        """
        QAT 전용 학습 설정.

        부모 클래스의 _setup_train()을 호출한 후,
        QAT에 필요한 설정을 오버라이드합니다.

        Args:
            world_size: 분산 학습 world size
        """
        # 부모 클래스의 _setup_train 호출 (기본 설정)
        super()._setup_train(world_size)

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
        모델 로드.

        부모 클래스의 get_model()을 그대로 사용하지만,
        QAT 모델이 이미 준비되어 있어야 합니다.

        Args:
            cfg: 모델 config
            weights: 가중치 경로
            verbose: 로그 출력 여부

        Returns:
            모델
        """
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        # TensorQuantizer 확인 로그
        try:
            from pytorch_quantization import nn as quant_nn
            quantizer_count = sum(1 for m in model.modules()
                                 if isinstance(m, quant_nn.TensorQuantizer))
            print(f"[QAT] 모델 로드 완료 - TensorQuantizer: {quantizer_count}개")
        except ImportError:
            pass

        return model

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

    def _do_train(self, world_size=1):
        """
        실제 학습 루프 수행.

        부모 클래스의 _do_train()을 그대로 사용합니다.
        QAT 설정은 _setup_train()에서 이미 적용되었습니다.

        Args:
            world_size: 분산 학습 world size
        """
        return super()._do_train(world_size=world_size)
