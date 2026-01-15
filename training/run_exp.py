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
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {args.config}")
    
    # --- Set Random Seed ---
    from src.utils import set_seed
    set_seed(config.get('seed', 42))
    print(f"Random seed set to: {config.get('seed', 42)}")
    
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
    
    # --- Configure Ultralytics Settings ---
    # Point 'weights_dir' to our pretrained_weights folder to avoid redundant downloads (e.g. AMP check)
    import ultralytics.settings as ul_settings
    # Use absolute path
    weights_dir = os.path.abspath(os.path.join(os.getcwd(), "pretrained_weights"))
    ul_settings.update({'weights_dir': weights_dir})
    print(f"Ultralytics weights_dir set to: {weights_dir}")

    print(f"Using data: {data_yaml}")
    
    trainer = PCBTrainer(model, config)
    
    # Initialize save_dir with a default in case training fails early
    save_dir = os.path.join("runs", config['exp_name']) 
    
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
