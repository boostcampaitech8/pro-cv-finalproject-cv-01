import argparse
import os
import sys
import yaml
import torch
import copy
from ultralytics import YOLO
from . import utils as qtu
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Hybrid EMA Re-calibration for QAT')
    parser.add_argument('--config', type=str, required=True, help='Path to config_qat.yaml')
    parser.add_argument('--weights', type=str, required=True, help='Path to best_qat.pt (QAT Checkpoint)')
    parser.add_argument('--output', type=str, default=None, help='Output path for hybrid model')
    return parser.parse_args()

def run_recalibration(config_path, weights_path, output_path=None):
    """
    Execute the EMA Re-calibration process programmatically.
    
    Args:
        config_path (str): Path to QAT configuration YAML.
        weights_path (str): Path to the trained checkpoint (best.pt).
        output_path (str, optional): Custom output path. Defaults to best_hybrid.pt.
    
    Returns:
        str: Path to the saved hybrid model.
    """
    print(f"\n🚀 Starting Hybrid EMA Re-calibration...")
    print(f"   Config: {config_path}")
    print(f"   Weights: {weights_path}")

    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Initialize QAT Context
    print("🔧 Initializing Quantization Modules...")
    qtu.initialize_quantization(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('device', 'cuda') != 'cpu' else 'cpu')
    print(f"💻 Device: {device}")

    # 3. Build Model Structure
    base_model_path = config['qat']['pretrained_path']
    print(f"🏗️  Building model skeleton from {base_model_path}...")
    model_wrapper = YOLO(base_model_path)
    model = model_wrapper.model
    
    # 4. Inject QAT Modules
    qtu.replace_with_quantization_modules(model)
    qtu.disable_sensitive_layers_quantization(model)
    model.to(device)

    # 5. Load EMA Weights
    print(f"📂 Loading Checkpoint from {weights_path}...")
    ckpt = torch.load(weights_path, map_location='cpu')
    
    ema_weights = None
    if isinstance(ckpt, dict):
        if 'ema' in ckpt and ckpt['ema'] is not None:
            print("✨ Found key 'ema' in checkpoint.")
            ema_obj = ckpt['ema']
            ema_weights = ema_obj
        else:
            print("⚠️  Key 'ema' not found. Checking 'model' key...")
            ema_weights = ckpt.get('model', ckpt)
    else:
        ema_weights = ckpt

    # Handle model object extraction
    if hasattr(ema_weights, 'model') and isinstance(ema_weights.model, torch.nn.Module):
         # Ultralytics Model wrapper
         print("📦 Found Ultralytics Model wrapper. Extracting inner .model...")
         ema_weights = ema_weights.model

    if hasattr(ema_weights, 'state_dict'):
        print("🔄 Converting EMA model object to state_dict...")
        try:
            ema_weights = ema_weights.state_dict()
        except Exception as e:
             print(f"⚠️  Could not call state_dict(): {e}")

    if ema_weights is None:
        print("❌ Critical Error: Failed to extract weights.")
        return None
        
    # Safe Loading (Prefix Handling)
    new_state_dict = OrderedDict()
    for k, v in ema_weights.items():
        # Do NOT strip 'model.' prefix blindly.
        # Check against target keys if needed, but here we assume DetectionModel expects 'model.'
        # Actually, let's trust the target model structure.
        new_state_dict[k] = v

    # Try loading as-is
    try:
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    except RuntimeError:
        # Fallback strip
        print("⚠️ Direct load failed. Trying to strip 'model.' prefix...")
        stripped = OrderedDict()
        for k, v in new_state_dict.items():
            if k.startswith('model.'):
                stripped[k[6:]] = v
            else:
                stripped[k] = v
        msg = model.load_state_dict(stripped, strict=False)
        print(f"✅ Fallback loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    
    # 6. Prepare Data for Calibration
    print("📚 Preparing Calibration Dataset...")
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from types import SimpleNamespace
    from ultralytics.utils import DEFAULT_CFG_DICT

    # Load data config
    data_yaml_path = config['data_path'] + '/data.yaml'
    if not os.path.exists(data_yaml_path):
        # Fallback to absolute path search or local
        data_yaml_path = 'PCB_DATASET/data.yaml' 
        
    data_cfg = check_det_dataset(data_yaml_path)
    
    dataset_path = data_cfg['train']
    if isinstance(dataset_path, list): dataset_path = dataset_path[0] 
    
    # Resolve relative path
    if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):
         # Try relative to config['data_path']
         dataset_path = os.path.join(config['data_path'], dataset_path)

    batch_size = config.get('batch_size', 8)
    img_size = config.get('img_size', 640)
    
    # Dummy CFG
    cfg_dict = DEFAULT_CFG_DICT.copy()
    cfg_dict.update({
        'imgsz': img_size,
        'rect': False,
        'stride': 32,
        'batch': batch_size,
        'task': 'detect',
        'data': data_cfg,
        'mode': 'train',
        'single_cls': False,
    })
    dummy_cfg = SimpleNamespace(**cfg_dict)

    dataset = build_yolo_dataset(
        cfg=dummy_cfg, 
        img_path=dataset_path, 
        batch=batch_size, 
        data=data_cfg, 
        mode='train', 
        rect=False, 
        stride=32
    )
    
    train_loader = build_dataloader(dataset, batch=batch_size, workers=4, shuffle=True, rank=-1)

    # 7. Re-Calibration Execution
    calib_method = config['qat']['calibration']['method']
    num_batches = config['qat']['calibration']['num_batches']
    
    print(f"⚖️  Running Re-Calibration on EMA Weights ({calib_method}, {num_batches} batches)...")
    qtu.collect_calibration_stats(model, train_loader, num_batches=num_batches, calib_method=calib_method, device=device)
    
    # 8. Save Hybrid Model
    if output_path is None:
        p_weights = Path(weights_path)
        # Suffix handling
        stem = p_weights.stem
        if 'qat' not in stem: stem += '_qat'
        output_path = str(p_weights.parent / f"{stem}_hybrid.pt")
        
    print(f"💾 Saving Hybrid QAT Model to {output_path}...")
    
    # Important: Save as FP32 to avoid export crash
    model.cpu().float()
    
    save_ckpt = {
        'model': model,
        'ema': None, 
        'optimizer': None,
        'info': 'Hybrid EMA+Calib Model',
        'date': str(os.path.getmtime(weights_path))
    }
    torch.save(save_ckpt, output_path)
    print("🎉 Recalibration Done!")
    
    return output_path

def main():
    args = get_args()
    run_recalibration(args.config, args.weights, args.output)

if __name__ == "__main__":
    main()
