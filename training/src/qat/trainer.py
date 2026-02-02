import os
import torch
import torch.optim as optim
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# Ultralytics Imports
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import unwrap_model

# Import QAT Utils (Relative)
from . import utils as qtu

class QATDetectionTrainer(DetectionTrainer):
    """
    Custom Trainer for Quantization-Aware Training (QAT).
    Inherits from Ultralytics DetectionTrainer to leverage EMA and robust training loops.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if cfg is None:
            cfg = DEFAULT_CFG_DICT.copy()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        
        # Default QAT Hyperparameters (can be overridden by args)
        self.qat_epochs = overrides.get('epochs', 20)
        self.qat_lr = overrides.get('lr0', 0.0001)

    def _setup_train(self, world_size=1):
        """
        Override setup to enforce QAT-specific constraints.
        """
        super()._setup_train()

        print("\n" + "="*40)
        print("🔧 QAT Specific Training Setup")
        print("="*40)

        # 1. Disable Warmup (Pretrained model doesn't need it)
        print(f"Original Warmup Epochs: {self.args.warmup_epochs}")
        self.args.warmup_epochs = 0
        print("👉 Forced Warmup Epochs: 0 (Disabled for QAT)")

        # 2. Disable AMP (Crucial for QAT Stability)
        print(f"Original AMP: {self.args.amp}")
        self.args.amp = False
        self.amp = False
        print("👉 Forced AMP: False (Disabled to prevent Q/DQ conflicts)")

        # 3. Custom cosine scheduler for fine-tuning
        # (Ultralytics one is good, but we want tighter control)
        # We replace the scheduler after super()._setup_train() initialized it
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.args.lr0 * 0.01
        )
        print("👉 Scheduler replaced with CosineAnnealingLR")
        print("="*40 + "\n")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Load model with strict=False to allow QAT layer mismatches if loading from checkpoint.
        In our pipeline, we usually inject the model externally, but this is a fallback.
        """
        # If model is already injected via self.model, super().get_model might still be called strictly.
        # But usually Ultralytics checks strict loading inside.
        
        # This fallback is mainly if we resume training
        model = super().get_model(cfg, weights, verbose)
        return model

    def save_model(self):
        """
        Override save_model to prevent .half() conversion.
        QAT models must remain in FP32 with QuantConv2d layers intact.
        """
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(unwrap_model(self.model)).float(),  # Force float32
            'ema': deepcopy(unwrap_model(self.ema.ema)).float() if self.ema else None, # Force float32
            'updates': self.ema.updates if self.ema else None,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),
            'date': datetime.now().isoformat(),
        }

        # Save to disk
        torch.save(ckpt, self.last)
        
        # Save best
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        
        # Periodic save
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')

        # [Debugging] Verify TensorQuantizer counts
        # from pytorch_quantization import nn as quant_nn
        # m = unwrap_model(self.model)
        # count = sum(1 for layer in m.modules() if isinstance(layer, quant_nn.TensorQuantizer))
        # print(f"[QAT Checkpoint] Saved with {count} quantizers intact.")
