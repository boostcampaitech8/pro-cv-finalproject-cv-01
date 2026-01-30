import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import sys
import os

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.qat import utils as qtu
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser(description='Export NVIDIA QAT Model to ONNX (Robust)')
    parser.add_argument('--weights', type=str, required=True, help='Path to best_qat.pt')
    parser.add_argument('--base-weights', type=str, required=True, help='Original FP32 best.pt for structure')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--output', type=str, default=None, help='Output ONNX path')
    parser.add_argument('--full-int8', action='store_true', help='Enable Head Quantization (Disable Hybrid Strategy)')
    parser.add_argument('--asymmetric', action='store_true', help='Use Asymmetric Quantization (Warn: TensorRT Mismatch)')
    parser.add_argument('--force-raw', action='store_true', help='Force use of Raw weights (model) even if EMA exists')
    return parser.parse_args()

def main():
    args = get_args()
    print("🚀 Starting NVIDIA QAT Export (Robust ver.)...")
    
    try:
        from pytorch_quantization import nn as quant_nn
    except ImportError:
        print("❌ pytorch-quantization not found.")
        return

    # 1. Initialize QAT Context
    print("🔧 Initializing Quantization Modules...")
    
    # Configure QAT based on arguments
    config = {'qat': {'quantization': {'weight_per_channel': True, 'num_bits': 8}, 'calibration': {'method': 'mse'}}}
    
    if args.asymmetric:
        print("⚠️  WARNING: Enabling Asymmetric Quantization (Legacy Mode). Expect TensorRT mismatch.")
        config['qat']['quantization']['symmetric'] = False
    else:
        print("✅ Symmetric Quantization Enabled (TensorRT Safe).")
        config['qat']['quantization']['symmetric'] = True

    qtu.initialize_quantization(config)

    # 2. Build Base Model using Ultralytics
    print(f"🏗️  Building base model from {args.base_weights}...")
    model_wrapper = YOLO(args.base_weights)
    model = model_wrapper.model
    
    # 3. Deep Recursive Replacement
    qtu.replace_with_quantization_modules(model)
    
    # [Hybrid Mode Check]
    if not args.full_int8:
        print("🛡️  Hybrid Mode: Disabling Head Quantization for stability...")
        qtu.disable_sensitive_layers_quantization(model)
    else:
        print("⚠️  Full INT8 Mode: Head Quantization ENABLED. (Sensitive Layers Active)")
    
    # 4. Load QAT State Dict (Handling keys)
    print(f"📂 Loading QAT weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location='cpu')
    
    # Robust extraction of state_dict
    state_dict = None
    if isinstance(checkpoint, dict):
        if args.force_raw and 'model' in checkpoint:
            print("📦 FORCE: Using 'model' (Raw) weights (Ignoring EMA per flag).")
            state_dict = checkpoint['model']
        elif 'ema' in checkpoint and checkpoint['ema'] is not None:
            print("✨ Found EMA weights. Using EMA for better accuracy.")
            state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            print("📦 Found model weights.")
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # If it's a model object instead of a state_dict
    if hasattr(state_dict, 'state_dict'):
        print("🔄 Converting model object to state_dict...")
        state_dict = state_dict.state_dict()
    
    # [FIX] Do NOT blindly strip 'model.' prefix.
    # Ultralytics DetectionModel expects keys starting with 'model.0...'
    # Only handle 'module.' wrapping (DDP).
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Try loading as-is first
    try:
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
    except RuntimeError:
        # Fallback: Try stripping 'model.' just in case target is Sequential
        print("⚠️ Direct load failed, trying to strip 'model.' prefix...")
        fallback_state_dict = OrderedDict()
        for k, v in new_state_dict.items():
             if k.startswith('model.'):
                 fallback_state_dict[k[6:]] = v
             else:
                 fallback_state_dict[k] = v
        msg = model.load_state_dict(fallback_state_dict, strict=False)
        print(f"⚠️ Fallback loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

    model.eval()
    model.float() # [FIX] Force FP32. QAT/ONNX export fails with FP16 weights.
    if torch.cuda.is_available(): model.cuda()

    # 5. Configure for ONNX
    print("⚙️  Configuring Detect Layer for ONNX...")
    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
            m.format = 'onnx'

    # 6. Force Enable Quantization (with safe amax handling)
    qtu.prepare_model_for_export(model)

    # 7. Execute Export
    if not args.output:
        args.output = args.weights.replace('.pt', '.onnx')
    
    # Robust Path Handling
    from pathlib import Path
    out_path = Path(args.output)
    if out_path.suffix != '.onnx':
        if out_path.exists() and out_path.is_dir():
             out_path = out_path / Path(args.weights).stem
        elif args.output.endswith('/'):
             out_path = out_path / Path(args.weights).stem
             
        out_path = out_path.with_suffix('.onnx')
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(out_path)
    
    print(f"📦 Exporting to {args.output} (Opset 13)...")
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz).to(next(model.parameters()).device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            opset_version=13,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
        )
        print("✅ ONNX Export successful.")

        # 8. Inject Metadata (CRITICAL for Ultralytics/TensorRT)
        print("💉 Injecting YOLO metadata into ONNX...")
        import onnx
        onnx_model = onnx.load(args.output)
        
        # Define metadata needed by Ultralytics
        meta_dict = {
            'names': eval(str(model.names)) if hasattr(model, 'names') else {i: f'class_{i}' for i in range(100)},
            'stride': int(max(model.stride.cpu().numpy())) if hasattr(model, 'stride') else 32,
            'task': 'detect',
            'batch': 1,
            'imgsz': [args.imgsz, args.imgsz]
        }
        
        # Helper to add metadata
        for k, v in meta_dict.items():
            meta = onnx_model.metadata_props.add()
            meta.key = k
            meta.value = str(v)
            
        onnx.save(onnx_model, args.output)
        print("✅ Metadata injected successfully.")
        
        # Verify
        print("\n🔍 Verifying ONNX Graph for Q/DQ nodes...")
        try:
            q_count = 0
            for node in onnx_model.graph.node:
                 if node.op_type == 'QuantizeLinear': q_count += 1
            
            if q_count > 0:
                print(f"🎉 SUCCESS: Found {q_count} QuantizeLinear nodes!")
            else:
                print("⚠️  WARNING: No Q/DQ nodes found in standard Opset 13 graph.")
                print("💡 Check if pytorch-quantization hooks were correctly triggered.")
        except ImportError:
            print("ℹ️  onnx library not installed, skipping graph verification.")

    except Exception as e:
        print(f"❌ Export failed: {e}")
    finally:
        quant_nn.TensorQuantizer.use_fb_fake_quant = False

if __name__ == "__main__":
    main()
