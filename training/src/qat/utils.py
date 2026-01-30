"""
QAT Utilities using NVIDIA pytorch-quantization toolkit.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from tqdm import tqdm

# Import Native Modules to avoid isinstance check failure after monkey-patching
from torch.nn import Conv2d as NativeConv2d
from torch.nn import Linear as NativeLinear

def initialize_quantization(config: Dict[str, Any]) -> None:
    """Initialize pytorch-quantization library and set default descriptors."""
    try:
        from pytorch_quantization import quant_modules
        from pytorch_quantization.tensor_quant import QuantDescriptor
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        raise ImportError("pytorch-quantization not installed.")

    # Suppress verbose logs from pytorch-quantization
    import logging
    logging.getLogger("pytorch_quantization.nn.modules.tensor_quantizer").setLevel(logging.ERROR)
    
    # Also suppress absl logging if possible (common in google-based libs)
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except ImportError:
        pass

    qat_config = config.get('qat', {})
    quant_config = qat_config.get('quantization', {})
    calibration_config = qat_config.get('calibration', {})
    
    quant_modules.initialize()

    num_bits = quant_config.get('num_bits', 8)
    weight_per_channel = quant_config.get('weight_per_channel', True)
    calib_method = calibration_config.get('method', 'mse')

    symmetric = quant_config.get('symmetric', True) # Default to True (TensorRT safe)

    if calib_method in ['mse', 'entropy', 'histogram']:
        input_desc = QuantDescriptor(num_bits=num_bits, calib_method='histogram')
    else:
        input_desc = QuantDescriptor(num_bits=num_bits, calib_method='max')

    if weight_per_channel:
        if symmetric:
            # [CRITICAL] TensorRT requires Symmetric Quantization (Zero Point = 0)
            weight_desc = QuantDescriptor(num_bits=num_bits, axis=(0,), unsigned=False, narrow_range=True)
        else:
            # Asymmetric (PyTorch Default) - Risk of TensorRT mismatch
            weight_desc = QuantDescriptor(num_bits=num_bits, axis=(0,))
    else:
        if symmetric:
            weight_desc = QuantDescriptor(num_bits=num_bits, unsigned=False, narrow_range=True)
        else:
            weight_desc = QuantDescriptor(num_bits=num_bits)

    quant_nn.QuantConv2d.set_default_quant_desc_input(input_desc)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(weight_desc)
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)

    print(f"✅ QAT Initialized: Bits={num_bits}, WeightPerChannel={weight_per_channel}, Calib={calib_method}")
    print(f"   Symmetric Weights: {symmetric} (TensorRT Compatible: {'YES' if symmetric else 'NO'})")

def replace_with_quantization_modules(model: nn.Module) -> None:
    """
    Manually replace nn.Conv2d and nn.Linear with QuantConv2d and QuantLinear.
    Robustly handles nn.Sequential, nn.ModuleList, and nested structures.
    Uses NativeConv2d to correctly identify layers even after monkey-patching.
    """
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    print("🔄 Manually replacing layers with Quantization modules (Deep Recursive)...")
    replaced_count = 0
    
    def _replace(module):
        nonlocal replaced_count
        
        # 1. Handle Sequential / ModuleList by iterating indices
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for i in range(len(module)):
                child = module[i]
                
                # Check using Native classes because nn.Conv2d might be monkey-patched already
                if isinstance(child, NativeConv2d) and not isinstance(child, quant_nn.QuantConv2d):
                    new_module = quant_nn.QuantConv2d(
                        child.in_channels, child.out_channels, child.kernel_size,
                        child.stride, child.padding, child.dilation, child.groups,
                        child.bias is not None, child.padding_mode
                    )
                    new_module.weight.data.copy_(child.weight.data)
                    if child.bias is not None: new_module.bias.data.copy_(child.bias.data)
                    module[i] = new_module
                    replaced_count += 1
                elif isinstance(child, NativeLinear) and not isinstance(child, quant_nn.QuantLinear):
                    new_module = quant_nn.QuantLinear(child.in_features, child.out_features, child.bias is not None)
                    new_module.weight.data.copy_(child.weight.data)
                    if child.bias is not None: new_module.bias.data.copy_(child.bias.data)
                    module[i] = new_module
                    replaced_count += 1
                else:
                    _replace(child)
            return

        # 2. Handle standard Modules by iterating named_children
        for name, child in module.named_children():
            if isinstance(child, NativeConv2d) and not isinstance(child, quant_nn.QuantConv2d):
                new_module = quant_nn.QuantConv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    child.stride, child.padding, child.dilation, child.groups,
                    child.bias is not None, child.padding_mode
                )
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None: new_module.bias.data.copy_(child.bias.data)
                setattr(module, name, new_module)
                replaced_count += 1
            elif isinstance(child, NativeLinear) and not isinstance(child, quant_nn.QuantLinear):
                new_module = quant_nn.QuantLinear(child.in_features, child.out_features, child.bias is not None)
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None: new_module.bias.data.copy_(child.bias.data)
                setattr(module, name, new_module)
                replaced_count += 1
            else:
                _replace(child)

    _replace(model)
    print(f"✅ Layer replacement complete. Replaced {replaced_count} layers.")

def disable_sensitive_layers_quantization(model: nn.Module) -> None:
    """Disable quantization for sensitive layers like Detect head and Attention blocks."""
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    # Patterns that often cause TensorRT fusion errors or significant mAP drop
    sensitive_patterns = ['detect', 'head', 'attn']
    
    print(f"🛡️  Disabling quantization for sensitive layers (Detect, Head, Attn)...")
    disabled_count = 0
    
    # helper to disable a specific module
    def _disable_module(m):
        did_disable = False
        if isinstance(m, (quant_nn.QuantConv2d, quant_nn.QuantLinear)):
            if hasattr(m, '_input_quantizer'):
                m._input_quantizer.disable()
                m._input_quantizer._explicitly_disabled = True
            if hasattr(m, '_weight_quantizer'):
                m._weight_quantizer.disable()
                m._weight_quantizer._explicitly_disabled = True
            did_disable = True
        return did_disable

    for name, module in model.named_modules():
        # 1. Check by Name
        if any(p in name.lower() for p in sensitive_patterns):
             if _disable_module(module): disabled_count += 1
             
        # 2. Check by Class Type (Crucial for YOLO Sequential models where name is just "23")
        # Check for Ultralytics Detect/Segment/Pose/Classify headers
        class_name = module.__class__.__name__.lower()
        if class_name in ['detect', 'segment', 'pose', 'classify', 'head']:
            print(f"   PLEASE NOTE: Found sensitive layer by type: {class_name} ({name})")
            # Recursively disable all children
            for child in module.modules():
                if _disable_module(child): disabled_count += 1

    print(f"✅ Disabled quantization for {disabled_count} sensitive modules.")

def collect_calibration_stats(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    num_batches: int = 64,
    calib_method: str = 'mse',
    device: str = 'cuda'
) -> None:
    """Run calibration to collect activation statistics."""
    try:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization import calib
    except ImportError:
        return

    print(f"📊 Starting Calibration ({calib_method} method, {num_batches} batches)...")
    model.eval()
    model.to(device)

    # 1. Enable Calibration
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if getattr(module, '_explicitly_disabled', False):
                continue
            
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()

    # 2. Feed Data
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Calibrating")):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                images = batch.get('img', batch.get('image'))
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device).float() / 255.0
            model(images)

    # 3. Disable Calibration & Compute Amax
    print(f"📉 Computing optimal amax stats using {calib_method}...")
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if getattr(module, '_explicitly_disabled', False):
                continue

            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
                
                if isinstance(module._calibrator, calib.HistogramCalibrator):
                    module.load_calib_amax(method=calib_method, strict=False)
                else:
                    module.load_calib_amax(strict=False)
                
                module._calibrator = None

    print("✅ Calibration Complete.")

def prepare_model_for_export(model: nn.Module) -> None:
    """Force enable quantization and set use_fb_fake_quant for all quantizers."""
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        return

    print("🔧 Forcing ONNX Export Mode (enable_quant + use_fb_fake_quant)...")
    
    # [CRITICAL] Set globally for the class to ensure hooks are registered correctly
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # Skip disabled modules (e.g. sensitive layers) to keep them pure FP32
            if getattr(module, '_explicitly_disabled', False):
                continue

            # [CRITICAL] Do NOT force enable_quant() here! 
            # It overrides 'disable_sensitive_layers_quantization', causing uncalibrated heads to be quantized.
            # module.enable_quant() 
            
            # Only set export flag
            module.use_fb_fake_quant = True
            
            # Check if amax exists; if not, assume it wasn't calibrated/quantized and keep as FP32
            if not hasattr(module, '_amax') or module._amax is None:
                 # Warning only, do not force
                 # print(f"⚠️  Warning: {name}._amax is None. Keeping as FP32 (Skipping Quantization).")
                 continue
            
            # Only set export flag if valid amax exists
            module.use_fb_fake_quant = True


def get_calibration_dataloader(
    data_yaml_path: str,
    imgsz: int = 640,
    batch_size: int = 8,
    workers: int = 4
):
    """
    Builds a PURE calibration dataloader with Augmentations DISABLED.
    This is critical for accurate statistic collection (amax).
    """
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.dataset import YOLODataset
    import yaml
    from ultralytics.utils import DEFAULT_CFG
    from types import SimpleNamespace
    
    # 1. Load Data Config
    data_cfg = check_det_dataset(data_yaml_path)
    train_path = data_cfg['train']
    if isinstance(train_path, list): train_path = train_path[0]
    
    # 2. Build Dataset directly (Bypassing Trainer to avoid default augs)
    # We use YOLODataset with augment=False
    # [FIX] DEFAULT_CFG is already an IterableSimpleNamespace, so use it directly
    hyp_obj = DEFAULT_CFG
    
    dataset = YOLODataset(
        img_path=train_path,
        imgsz=imgsz,
        batch_size=batch_size,
        augment=False,       # <--- CRITICAL: Disable Mosaic/Mixup
        hyp=hyp_obj,         # <--- Fix: Pass object wrapper
        rect=False,          # Standard square inference-like
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        data=data_cfg,
        classes=data_cfg['names']
    )
    
    # 3. Build Loader
    from torch.utils.data import DataLoader
    # Use Ultralytics collate_fn
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,        # Shuffle is fine, but data itself is clean
        num_workers=workers,
        collate_fn=getattr(dataset, 'collate_fn', None),
        pin_memory=True
    )
    
    return loader
