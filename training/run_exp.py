import yaml
import argparse
from src.train import PCBTrainer
from src.inference import InferenceMgr
import os
import importlib 

def main():
    parser = argparse.ArgumentParser(description="PCB Defect Detection Experiment Runner")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load Config
    config_path = args.config
    if not os.path.exists(config_path):
        # Fallback: Try finding it relative to the script directory
        # This handles cases where user runs from root like: python training/run_exp.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_path = os.path.join(script_dir, config_path)
        if os.path.exists(candidate_path):
            print(f"Config '{config_path}' not found in CWD. Found at '{candidate_path}'.")
            config_path = candidate_path
            
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {args.config}")

    # --- Path Resolution (Relative -> Absolute) ---
    # Resolve 'data_path' relative to this script's location (training/)
    # This ensures portability regardless of where command is run.
    if 'data_path' in config and not os.path.isabs(config['data_path']):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path = os.path.abspath(os.path.join(base_dir, config['data_path']))
        print(f"Resolving relative data_path '{config['data_path']}' -> '{resolved_path}'")
        config['data_path'] = resolved_path
    
    # --- Set Random Seed ---
    from src.utils import set_seed, setup_logging, increment_path
    set_seed(config.get('seed', 42))
    print(f"Random seed set to: {config.get('seed', 42)}")
    
    # --- Configure Ultralytics Settings ---
    # Point 'weights_dir' to our pretrained_weights folder to avoid redundant downloads (e.g. AMP check)
    from ultralytics import settings as ul_settings
    # Use absolute path
    # Weights dir is sibling to src, assuming run_exp.py is in training/
    # If run_exp.py is in training/, os.getcwd() might be anything.
    # Better to use script location base.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.abspath(os.path.join(base_dir, "pretrained_weights"))
    
    ul_settings.update({'weights_dir': weights_dir})
    print(f"Ultralytics weights_dir set to: {weights_dir}")
    
    # --- Dynamic Loading ---
    
    # 1. Dataset Selection
    ds_mod_name = config.get('dataset_module', 'dataset')
    print(f"Loading Dataset Module: src/datasets/{ds_mod_name}.py")
    try:
        ds_lib = importlib.import_module(f"src.datasets.{ds_mod_name}")
        dataset = ds_lib.get_dataset(config)
    except ImportError as e:
        print(f"Error loading dataset module: {e}")
        return
    except AttributeError:
        print(f"Module src.datasets.{ds_mod_name} must have a 'get_dataset(config)' function.")
        return

    # Prepare Data
    data_yaml = dataset.prepare()

    # 2. Model Selection
    md_mod_name = config.get('model_module', 'yolov8')
    print(f"Loading Model Module: src/models/{md_mod_name}.py")
    
    try:
        md_lib = importlib.import_module(f"src.models.{md_mod_name}")
        # Pass entire config to the model factory
        model = md_lib.get_model(config)
    except ImportError as e:
        print(f"Error loading model module: {e}")
        return
    except AttributeError:
        print(f"Module src.models.{md_mod_name} must have a 'get_model(config)' function.")
        return

    # 3. Training
    print(f"\n{'='*20} Start Training (Exp: {config['exp_name']}) {'='*20}")
    
    print(f"Random seed set to: {config.get('seed', 42)}")
    

    print(f"Using data: {data_yaml}")
    
    trainer = PCBTrainer(model, config)
    
    # Initialize save_dir (using 'project' from config if set, else relative 'runs')
    # Force 'runs' to be in the same directory as this script (training/)
    # This ensures consistency even if run from root.
    runs_dir = os.path.join(base_dir, "runs")
    config['project'] = runs_dir
    print(f"Set save project directory to: {runs_dir}")
    
    # --- Manual Directory Creation & Logging Setup ---
    # We want to capture logs, so we need to know the folder BEFORE training starts.
    # We manually handle the 'increment_path' logic here.
    
    initial_exp_name = config['exp_name']
    base_save_dir = os.path.join(runs_dir, initial_exp_name)
    
    # Find a free directory (e.g. runs/baseline -> runs/baseline2)
    save_dir = increment_path(Path(base_save_dir), exist_ok=False, mkdir=True)
    save_dir = str(save_dir) # Convert to string
    
    # Update config with the ACTUAL name (e.g. 'baseline2') so YOLO uses this folder
    # We extract the folder name from the full path
    actual_exp_name = os.path.basename(save_dir)
    config['exp_name'] = actual_exp_name
    
    # Setup Logging to 'runs/exp_name/console.log'
    setup_logging(save_dir)
    
    try:
        # Train
        actual_save_dir = trainer.train(data_yaml)
        if actual_save_dir:
            save_dir = str(actual_save_dir)

        print("\nTraining completed.")

        # 3. Final Inference on Test Set
        # Model should be at {save_dir}/weights/best.pt
        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        
        print(f"Running inference on Test Set using model: {best_model_path}")
        
        if os.path.exists(best_model_path):
            inference_mgr = InferenceMgr(best_model_path, config)
            test_list_path = os.path.join(config['data_path'], 'test_images.txt')
            
            if os.path.exists(test_list_path):
                # Output to {save_dir}/inference
                inference_mgr.predict(test_list_path, project=save_dir, name="inference")
        
        else:
            print(f"Best model not found at {best_model_path}")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup / Organization Logic ---
        # Delegate to src/utils.py
        from src.utils import cleanup_artifacts
        cleanup_artifacts(save_dir, config)

if __name__ == "__main__":
    main()
