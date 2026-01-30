import os
import yaml
import subprocess
import time
from itertools import product
import sys

# Configuration
MODELS = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov11n', 'yolov11s', 'yolov11m']
IMG_SIZES = [640, 960, 1280, 1600]
BASE_CONFIG_PATH = '../configs/config.yaml'

def main():
    # Ensure we are in the script directory to find config.yaml and run_exp.py easily
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 1. Load Base Config
    if not os.path.exists(BASE_CONFIG_PATH):
        print(f"Error: {BASE_CONFIG_PATH} not found in {os.getcwd()}")
        return

    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    # 2. Iterate
    total_exps = len(MODELS) * len(IMG_SIZES)
    current_idx = 0

    print(f"Found {len(MODELS)} models: {MODELS}")
    print(f"Found {len(IMG_SIZES)} resolutions: {IMG_SIZES}")
    print(f"Total Scheduled Experiments: {total_exps}")
    print("=" * 60)
    
    for model, img_size in product(MODELS, IMG_SIZES):
        current_idx += 1
        exp_name = f"{model}_{img_size}"
        
        print(f"\n[{current_idx}/{total_exps}] Starting Experiment: {exp_name} (Model: {model}, Img: {img_size})")
        print("-" * 60)
        
        # Modify Config
        exp_config = base_config.copy()
        exp_config['model_module'] = model
        exp_config['img_size'] = img_size
        exp_config['exp_name'] = exp_name
        
        # Create Temp Config Name
        temp_config_path = f"config_{exp_name}.yaml"
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(exp_config, f)
            
        try:
            # Run Experiment
            # We use the current python executable
            cmd = [sys.executable, "run_exp.py", "--config", temp_config_path]
            
            start_t = time.time()
            subprocess.run(cmd, check=True)
            duration = time.time() - start_t
            
            print(f"✅ Experiment {exp_name} finished in {duration/60:.1f} mins.")
            
        except subprocess.CalledProcessError:
            print(f"❌ Experiment {exp_name} FAILED. Check logs. Continuing to next...")
        except KeyboardInterrupt:
            print("\n⚠️ Scheduling interrupted by user. Cleaning up and exiting...")
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            sys.exit(1)
        finally:
            # Cleanup Temp Config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    print("\n" + "=" * 60)
    print("🎉 All scheduled experiments completed.")

if __name__ == "__main__":
    main()
