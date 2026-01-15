import os
import shutil

def cleanup_artifacts(save_dir, config):
    """
    Cleans up stray directories and temporary files after training.
    Moves 'runs/detect' to '{save_dir}/detect' and removes AMP check files.
    """
    print("\nRunning Cleanup Logic...")
    
    # Target directory for moved detection results
    target_detect = os.path.join(save_dir, "detect")
    
    # List of stray items to look for
    # 1. 'runs/detect' (Created by YOLO defaults sometimes)
    stray_detect_runs = os.path.join("runs", "detect")
    # 2. 'detect' (Created in root if project path fails)
    stray_detect_root = "detect"
    # 3. AMP check artifacts
    stray_amp_model_11 = "yolo11n.pt"
    stray_amp_model_8 = "yolov8n.pt"

    strays = [stray_detect_runs, stray_detect_root, stray_amp_model_11, stray_amp_model_8]

    for stray in strays:
        if os.path.exists(stray):
            # Case A: File (AMP artifacts) -> Delete
            if os.path.isfile(stray):
                try:
                    os.remove(stray)
                    print(f"Cleanup: Removed temporary file '{stray}' (AMP check artifact).")
                except Exception as e:
                    print(f"Failed to remove '{stray}': {e}")
                continue

            # Case B: Directory (Detect results) -> Move
            print(f"Notice: Stray '{stray}' found. Moving to {target_detect}...")
            try:
                os.makedirs(save_dir, exist_ok=True)
                
                # Check if target exists
                if not os.path.exists(target_detect):
                    shutil.move(stray, target_detect)
                else:
                    # Merge contents if target already exists
                    for item in os.listdir(stray):
                        s = os.path.join(stray, item)
                        d = os.path.join(target_detect, item)
                        if os.path.exists(d):
                            if os.path.isdir(d): shutil.rmtree(d)
                            else: os.remove(d)
                        shutil.move(s, d)
                    shutil.rmtree(stray)
                print(f"Moved '{stray}' successfully.")
            except Exception as cleanup_e:
                print(f"Failed to move '{stray}': {cleanup_e}")

def set_seed(seed=42):
    """
    Sets random seeds for reproducibility.
    (Note: YOLOv8 handles this internally via 'seed' arg, 
     but this is good for other random operations if any).
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
