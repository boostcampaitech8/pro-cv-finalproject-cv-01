import os
import sys
import yaml
import argparse
import sys
import os

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import socket
# [MONKEY PATCH] Bypass Ultralytics online check hang
_original_getaddrinfo = socket.getaddrinfo
def _dummy_getaddrinfo(*args, **kwargs):
    # Retrieve original results if it's localhost (optional)
    # But for now, just simulate offline or instant failure to fallback
    raise socket.gaierror("Simulated offline for speed")
socket.getaddrinfo = _dummy_getaddrinfo

# Import Ultralytics parts
try:
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
finally:
    # Restore
    socket.getaddrinfo = _original_getaddrinfo

# Import Custom QAT Utils
from src.qat import utils as qtu
from src.qat import recalibrate

def get_args():
    parser = argparse.ArgumentParser(description='NVIDIA TensorRT-Optimized QAT for YOLOv8')
    parser.add_argument('--config', type=str, default='configs/config_qat.yaml', help='Path to QAT config file')
    return parser.parse_args()

def main():
    args = get_args()
    print("🚀 Starting NVIDIA QAT Training Pipeline...")
    
    # 1. Load Config
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        return
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"📄 Loaded config from {args.config}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Setup Device
    device_str = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # Handle '0' -> 'cuda:0'
    if device_str.isdigit():
        device_str = f"cuda:{device_str}"
    device = torch.device(device_str)
    print(f"💻 Device: {device}")

    # 2. Initialize QAT (NVIDIA Auto-Injection)
    # MUST be done before loading model to auto-replace layers
    print("🔧 Initializing NVIDIA pytorch-quantization...")
    qtu.initialize_quantization(config)

    # 3. Load Model
    pretrained_path = config['qat']['pretrained_path']
    if not os.path.isabs(pretrained_path):
        pretrained_path = os.path.join(project_root, pretrained_path)
    
    print(f"📦 Loading pretrained weights: {pretrained_path}")
    if not os.path.exists(pretrained_path):
        print(f"❌ Weights file not found: {pretrained_path}")
        return

    # YOLO wrapper to load weights
    yolo_wrapper = YOLO(pretrained_path) 
    model = yolo_wrapper.model
    
    # [CRITICAL] Force replace layers because Ultralytics loading might bypass auto-patching
    print("🔄 Forcing Deep Layer Replacement for QAT Injection...")
    qtu.replace_with_quantization_modules(model)
    
    model.to(device).float().train()

    # Disable Q for sensitive layers (Head, Attention)
    # 4. Prepare Data Loaders (Clean Calibration)
    data_path = config.get('data_path', './PCB_DATASET')
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)
    data_yaml = os.path.join(data_path, 'data.yaml')

    imgsz = config.get('img_size', 640)
    batch_size = config.get('batch_size', 8)
    workers = config.get('workers', 4)

    print(f"📚 Preparing Clean Calibration Data (imgsz={imgsz}, batch={batch_size})...")
    # Use the new helper to get Agumentation-Free loader
    train_loader = qtu.get_calibration_dataloader(
        data_yaml_path=data_yaml,
        imgsz=imgsz,
        batch_size=batch_size,
        workers=workers
    )

    # Hack: We still need a trainer later for fine-tuning, but we init it later.

    # [Full Quantization Strategy] -> Reverted to Hybrid due to v11m sensitivity (Class 4 mAP 0.25)
    # Even with clean calibration, explicit Head quantization hurts small classes in v11.
    if config['qat']['quantization'].get('skip_last_layers', True):
        print("🛡️  Hybrid Mode: Disabling Head Quantization for stability...")
        qtu.disable_sensitive_layers_quantization(model)
    else:
        print("⚡ Full Quantization Mode: Head Quantization Enabled")

    # 5. Calibration (MSE)
    # Critical step for mAP
    calib_batches = config['qat']['calibration'].get('num_batches', 64)
    calib_method = config['qat']['calibration'].get('method', 'mse')
    
    print(f"\n⚖️  Running Calibration ({calib_method})...")
    qtu.collect_calibration_stats(model, train_loader, num_batches=calib_batches, calib_method=calib_method, device=device)
    # [CRITICAL] Force model to device again. load_calib_amax might have set amax on CPU.
    model.to(device)
    print("✅ Calibration finished. Optimal scales loaded.")

    # 6. Fine-Tuning with Custom Trainer (EMA, AMP Off, Warmup Off)
    print("\n▶️  Starting QAT Fine-tuning with Ultralytics Trainer...")
    
    # Import Custom Trainer
    from src.qat.trainer import QATDetectionTrainer
    
    # Prepare Trainer Arguments
    args_dict = {
        'model': pretrained_path, # Dummy for init, we will inject model
        'data': data_yaml,
        'epochs': config['qat']['finetune']['epochs'],
        'imgsz': imgsz,
        'batch': batch_size,
        'workers': workers,
        'device': device_str,
        'optimizer': 'AdamW',
        'lr0': float(config['qat']['finetune']['lr0']),
        'lrf': 0.01,
        'weight_decay': config.get('weight_decay', 0.0005),
        'project': os.path.join(project_root, config.get('project', 'runs/qat')),
        'name': config.get('exp_name', 'default'),
        'exist_ok': True,
        'val': True,
        'plots': False,      # Disable plot generation to save time/output
        'verbose': False,    # [CLEANUP] Disable per-class metric table
        # Loss gains
        'box': config.get('box', 7.5),
        'cls': config.get('cls', 0.5),
        'dfl': config.get('dfl', 1.5),
        # Augmentations (Inherit from config or defaults)
        'hsv_h': config.get('hsv_h', 0.015),
        'hsv_s': config.get('hsv_s', 0.7),
        'hsv_v': config.get('hsv_v', 0.4),
    }
    
    # Initialize Trainer
    trainer = QATDetectionTrainer(overrides=args_dict)
    
    # [CRITICAL] Inject our QAT-Initialized Model
    # Since the trainer might try to load 'model' path, we overwrite it immediately
    trainer.model = model
    
    # Start Training
    train_results = trainer.train()
    
    print("🎉 QAT Training Pipeline Completed!")
    
    # Final save path logic is handled by Trainer, but we can print it
    print(f"💾 Checkpoints saved to: {trainer.save_dir}")

    # 7. Automated Recalibration (One-Click)
    print("\n🔗 Automatically triggering EMA Recalibration...")
    best_pt_path = str(trainer.best)
    if os.path.exists(best_pt_path):
        hybrid_path = recalibrate.run_recalibration(args.config, best_pt_path)
        print(f"✅ Full QAT Pipeline Finished. Ready for Export: {hybrid_path}")
    else:
        print(f"⚠️ Could not find best.pt at {best_pt_path}. Skipping recalibration.")

if __name__ == "__main__":
    main()
