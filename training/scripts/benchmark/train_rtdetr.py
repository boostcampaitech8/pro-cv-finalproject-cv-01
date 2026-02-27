from ultralytics import RTDETR
import os
import argparse

def train_rtdetr(model_name='rtdetr-l.pt', epochs=100, batch=16, imgsz=640, device='0', patience=20):
    """
    Train RT-DETR model using Ultralytics.
    RT-DETR uses 640x640 by default, but we can try 960 if supported.
    Usually RT-DETR works well with standard 640.
    """
    print(f"[Benchmark] RT-DETR Training: {model_name}")
    
    # Load model
    model = RTDETR(model_name)
    
    # Project path
    save_dir = '/data/ephemeral/home/final_project/training/runs/benchmark'
    
    # Train
    results = model.train(
        data='PCB_DATASET/data.yaml',
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=save_dir,
        name=f"rtdetr_{imgsz}",
        exist_ok=True,
        seed=0,
        deterministic=True,
        patience=patience
    )
    
    # Validate
    metrics = model.val(
        data='PCB_DATASET/data.yaml', 
        split='test',
        project=save_dir,
        name=f"rtdetr_{imgsz}_eval"
    )
    
    print(f"[Results] RT-DETR {model_name} mAP50: {metrics.box.map50}")
    print(f"[Results] RT-DETR {model_name} mAP50-95: {metrics.box.map}")
    
    # Calculate FPS
    # metrics.speed = {'preprocess': X, 'inference': Y, 'postprocess': Z} (ms)
    speed = metrics.speed
    total_time_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
    fps = 1000.0 / total_time_ms
    print(f"[Results] RT-DETR {model_name} Speed: {total_time_ms:.2f}ms/img ({fps:.2f} FPS)")

    # Log to CSV
    try:
        from benchmark_utils import log_to_csv
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from benchmark_utils import log_to_csv

    # Extract Class-wise AP (metrics.box.maps is array of AP50-95 per class)
    # We want AP50? Ultralytics stores map50 per class? 
    # Actually metrics.box.maps is mAP50-95 per class usually. 
    # Let's verify: metrics.box.maps is (N,) array.
    # We will log mAP50-95 per class as our "Class AP".
    
    # Class Names (Hardcoded or loaded?)
    # Ultralytics model has names
    names = model.names # dict {0: 'missing_hole'...}
    class_aps = metrics.box.maps # Array of mAP50-95 per class
    
    # Create Result Dict
    log_data = {
        'Model': f"RT-DETR ({model_name})",
        'Resolution': imgsz,
        'Patience': patience,
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'Speed(ms)': total_time_ms,
        'FPS': fps
    }
    
    # Add Class APs
    for i, ap in enumerate(class_aps):
        # Clean class name for CSV header (matches benchmark_utils fields)
        cls_name = names[i] + '_AP' 
        log_data[cls_name] = ap

    log_to_csv(log_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rtdetr-l.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()
    
    train_rtdetr(model_name=args.model, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device, patience=args.patience)
