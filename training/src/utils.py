import os
import shutil
import sys
import glob
import re
from pathlib import Path

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

class Tee(object):
    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    
    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush() # Ensure immediate flush
        self.stdout.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp -> runs/exp{sep}2, runs/exp{sep}3, ...
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        
        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                path = Path(p)
                break
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
        
    return path

def setup_logging(save_dir):
    """
    Sets up a Tee logger to mirror stdout/stderr to a file in save_dir.
    """
    log_path = os.path.join(save_dir, "console.log")
    # Prevent double logging if called multiple times or reloaded
    if not isinstance(sys.stdout, Tee):
        Tee(log_path)
        print(f"Logging initialized. Output is being saved to: {log_path}")
