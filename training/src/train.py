class PCBTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, data_yaml):
        """
        Executes training for a specific fold data.yaml
        """
        import wandb
        import mlflow
        
        # MLflow Setup
        # 1. Disable Ultralytics Auto-MLflow (Prevents conflicts)
        from ultralytics import settings as ul_settings
        ul_settings.update({'mlflow': False})
        
        # 2. Configure Local MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("PCB_Res_Experiments")
        
        # WandB Setup
        if self.config.get('wandb_project'):
            wandb.init(project=self.config['wandb_project'], name=f"{self.config['exp_name']}_fold", reinit=True)
            
        # Scheduler Logic
        # YOLOv8 uses 'cos_lr' argument: True=Cosine, False=Linear
        cos_lr = True if self.config.get('scheduler_type') == 'cosine' else False
        
        # Optimizer Setup (passing explicit params)
        # Note: YOLOv8 trainer handles optimizer creation, we just pass the name and params.
        
        # Add Custom Callback for Clean Logging
        # Note: We reverted to default YOLO logging to show TQDM bar
        from ultralytics.utils import LOGGER
        import logging
        import time
        
        # Custom Progress Bar Logic
        import sys

        # Suppress generic YOLO info logs (Scanning dataset...)
        LOGGER.setLevel(logging.WARNING)

        # Disable Default Ultralytics MLflow Callback (prevents Run 'not found' errors)
        # Ultralytics tries to auto-log if mlflow is installed. We handle it manually.
        from ultralytics.utils.callbacks import mlflow as mlflow_callbacks
        
        # Safely remove MLflow callbacks if they exist
        checks = [
            ('on_pretrain_routine', 'on_pretrain_routine'),
            ('on_fit_epoch_end', 'on_fit_epoch_end'),
            ('on_train_end', 'on_train_end')
        ]
        
        for cb_name, list_name in checks:
            try:
                cb_func = getattr(mlflow_callbacks, cb_name, None)
                if cb_func and cb_func in self.model.callbacks.get(list_name, []):
                    self.model.callbacks[list_name].remove(cb_func)
            except Exception:
                pass
        
        epoch_start_time = 0
        current_batch_idx = 0

        def on_train_epoch_start(trainer):
            nonlocal epoch_start_time, current_batch_idx
            epoch_start_time = time.time()
            current_batch_idx = 0 # Reset batch counter

        def on_train_batch_end(trainer):
            """Manual Progress Bar: Prints on the same line using \\r"""
            nonlocal current_batch_idx
            current_batch_idx += 1
            
            current_epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            current_batch = current_batch_idx
            # Access total batches correctly
            total_batches = len(trainer.train_loader)
            
            # Loss items (box, cls, dfl) - typically a tensor of shape (3,)
            # We check if it exists and pull values
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                # Format loss nicely
                losses = [f"{x.item():.4f}" for x in trainer.loss_items]
                loss_str = f"box: {losses[0]} cls: {losses[1]} dfl: {losses[2]}"
            else:
                loss_str = "loss: N/A"

            # Progress Bar UI
            percent = int((current_batch / total_batches) * 100)
            bar_len = 20
            filled_len = int(bar_len * current_batch // total_batches)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            
            # Print with \r (Carriage Return)
            # Epoch 1/200 [████------] 40% | box: 1.2 cls: 0.5 ...
            sys.stdout.write(f"\rEpoch {current_epoch}/{total_epochs} [{bar}] {percent}% | {loss_str}")
            sys.stdout.flush()

        def on_fit_epoch_end(trainer):
            """End of Epoch Summary"""
            nonlocal epoch_start_time
            metrics = trainer.metrics
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            duration = time.time() - epoch_start_time
            
            # Clear the progress bar line by overwriting with spaces or just print newline
            # But we want to preserve the final state? 
            # Let's print the clean validation summary on a NEW line so the bar stays above (or we overwrite it).
            # User prefers cleanliness. Let's overwrite the bar with the final summary.
            
            # Get mAP metrics
            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
            
            # Calculate YOLO Fitness: 0.1 * mAP50 + 0.9 * mAP50-95
            fitness = (0.1 * map50) + (0.9 * map50_95)
            
            # Extract losses (from last batch of the epoch)
            if hasattr(trainer, 'loss_items'):
                box_loss = trainer.loss_items[0].item()
                cls_loss = trainer.loss_items[1].item()
                dfl_loss = trainer.loss_items[2].item()
            else:
                box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0

            # Print with Fitness
            print(f"Epoch {epoch}/{total_epochs} | box: {box_loss:.4f} cls: {cls_loss:.4f} dfl: {dfl_loss:.4f} | mAP50: {map50:.4f} | Fitness: {fitness:.4f} | Time: {duration:.2f}s")
            
            # MLflow Metric Logging (Per Epoch)
            if mlflow.active_run():
                mlflow.log_metrics({
                    "train/box_loss": box_loss,
                    "train/cls_loss": cls_loss,
                    "train/dfl_loss": dfl_loss,
                    "metrics/mAP50": map50,
                    "metrics/mAP50-95": map50_95,
                    "metrics/fitness": fitness
                }, step=epoch)
            
            # Print Class-wise mAP50 if available
            try:
                if hasattr(trainer, 'validator') and trainer.validator is not None:
                    # Check if metrics exist (might be None on first few epochs if val skipped)
                    if hasattr(trainer.validator, 'metrics') and trainer.validator.metrics is not None:
                        # trainer.validator.metrics.ap_class_index is a list of existing class indices
                        # trainer.validator.metrics.class_result(i) returns (p, r, ap50, ap)
                        
                        metrics = trainer.validator.metrics
                        names = trainer.validator.names
                        
                        # ap_class_index can be a tensor or list
                        if hasattr(metrics, 'ap_class_index') and len(metrics.ap_class_index) > 0:
                            print(f"{'Class':<20} | {'mAP50':<10}")
                            print("-" * 35)
                            for i, cls_idx in enumerate(metrics.ap_class_index):
                                cls_idx = int(cls_idx)
                                name = names.get(cls_idx, str(cls_idx))
                                # class_result(i) returns results for the i-th class in ap_class_index
                                # Note: Check signature of class_result. usually it takes index in ap_class_index, or index of sorted api?
                                # Actually class_result(i) takes the index in the *sorted* list usually.
                                # Let's try to access metrics.ap directly if possible, but class_result is safer helper.
                                # In Ultralytics v8.1+, class_result(i) returns (p[i], r[i], ap50[i], ap[i])
                                
                                res = metrics.class_result(i)
                                map50_cls = res[2]
                                print(f"{name:<20} | {map50_cls:.4f}")
                            print("-" * 35)
            except Exception as e:
                # Fallback if internal API changes or access fails
                # print(f"Could not print class metrics: {e}")
                pass

        # Register Callbacks
        self.model.add_callback("on_train_epoch_start", on_train_epoch_start)
        self.model.add_callback("on_train_batch_end", on_train_batch_end)
        self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # Start MLflow Run
        if mlflow.active_run():
            mlflow.end_run()
            
        mlflow.start_run(run_name=self.config['exp_name'])
        mlflow.log_params(self.config)

        try:
            results = self.model.train(
            data=data_yaml,
            epochs=self.config['epochs'],
            batch=self.config['batch_size'],
            imgsz=self.config['img_size'],
            patience=self.config['patience'],
            save=self.config['save'],
            save_period=self.config['save_period'],
            device=self.config['device'],
            workers=self.config['workers'],
            
            project=self.config.get('project', 'runs'),
            name=self.config['exp_name'], # Custom Experiment Name
            exist_ok=True, 
            
            # Optimizer & Scheduling
            optimizer=self.config.get('optimizer', 'auto'),
            lr0=self.config.get('lr0', 0.01),
            lrf=self.config.get('lrf', 0.01),
            momentum=self.config.get('momentum', 0.937),
            weight_decay=self.config.get('weight_decay', 0.0005),
            cos_lr=cos_lr,
            
            # Augmentation (Controlled via config.yaml)
            hsv_h=self.config.get('hsv_h', 0.015),
            hsv_s=self.config.get('hsv_s', 0.7),
            hsv_v=self.config.get('hsv_v', 0.4),
            degrees=self.config.get('degrees', 0.0),
            translate=self.config.get('translate', 0.1),
            scale=self.config.get('scale', 0.5),
            shear=self.config.get('shear', 0.0),
            perspective=self.config.get('perspective', 0.0),
            flipud=self.config.get('flipud', 0.0),
            fliplr=self.config.get('fliplr', 0.5),
            mosaic=self.config.get('mosaic', 1.0),
            mixup=self.config.get('mixup', 0.0),
            copy_paste=self.config.get('copy_paste', 0.0),
            
            # Visualization
            plots=True, # Enable JPG saving
            
            # Loss Gains
            box=self.config.get('box', 7.5),
            cls=self.config.get('cls', 0.5),
            dfl=self.config.get('dfl', 1.5),
            
            verbose=False,  # Disable default TQDM bar (We are using manual)
            val=True       # Enable validation
        )
            
        except Exception as e:
            print(f"Training failed: {e}")
            if mlflow.active_run():
                mlflow.end_run()
            raise e
        
        # Run Final Evaluation on Best Model
        self.run_final_eval(data_yaml)
        
        # Return the actual save directory (Path object) to handle auto-increment (e.g. baseline2)
        return self.model.trainer.save_dir

    def run_final_eval(self, data_yaml):
        """
        Loads the best model and runs a final validation to print comprehensive metrics.
        """
        import os
        from ultralytics import YOLO
        
        # Ensure trainer and save_dir exist
        if not hasattr(self.model, 'trainer') or not self.model.trainer:
            print("Trainer not initialized, skipping final eval.")
            return

        save_dir = self.model.trainer.save_dir
        # YOLOv8 default save structure: weights/best.pt
        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        
        if not os.path.exists(best_model_path):
            print(f"Warning: Best model not found at {best_model_path}. Skipping final evaluation.")
            return

        print(f"\n{'='*20} FINAL EVALUATION (Best Model) {'='*20}")
        print(f"Loading best model from: {best_model_path}")
        
        try:
            # Load best model
            best_model = YOLO(best_model_path)
            
            # Run validation
            # verbose=True ensures standard YOLO table is also printed if we miss something
            print("Running validation on best model...")
            metrics = best_model.val(data=data_yaml, split='val', verbose=False)
            
            print("\n[Best Model Class-wise Performance]")
            print(f"{'Class':<20} | {'mAP50':<10} | {'mAP50-95':<10}")
            print("-" * 50)
            
            names = metrics.names
            if hasattr(metrics, 'ap_class_index'):
                for i, cls_idx in enumerate(metrics.ap_class_index):
                    cls_idx = int(cls_idx)
                    name = names.get(cls_idx, str(cls_idx))
                    
                    # metrics.class_result(i) -> (p, r, map50, map50-95)
                    # This relies on Ultralytics implementation details
                    try:
                        res = metrics.class_result(i)
                        map50 = res[2]
                        map50_95 = res[3]
                        print(f"{name:<20} | {map50:.4f}     | {map50_95:.4f}")
                    except Exception:
                        # Fallback if structure differs
                        print(f"{name:<20} | N/A        | N/A")
                        
            print("-" * 50)
            print(f"Overall mAP50: {metrics.box.map50:.4f}")
            print(f"Overall mAP50-95: {metrics.box.map:.4f}")
            print(f"{'='*50}\n")
            
            # Log Final Best Metrics to MLflow
            if mlflow.active_run():
                # 1. Global Metrics
                log_data = {
                    "final_mAP50": metrics.box.map50,
                    "final_mAP50-95": metrics.box.map
                }
                
                # 2. Class-wise Metrics
                try:
                    names = metrics.names
                    if hasattr(metrics, 'ap_class_index'):
                        for i, cls_idx in enumerate(metrics.ap_class_index):
                            cls_idx = int(cls_idx)
                            name = names.get(cls_idx, str(cls_idx))
                            
                            # Clean name for MLflow (no spaces/special chars)
                            safe_name = name.replace(" ", "_")
                            
                            res = metrics.class_result(i)
                            map50_cls = res[2]
                            map50_95_cls = res[3]
                            
                            log_data[f"final_mAP50_{safe_name}"] = map50_cls
                            log_data[f"final_mAP95_{safe_name}"] = map50_95_cls
                except Exception as e:
                    print(f"Warning: Could not log class metrics: {e}")

                mlflow.log_metrics(log_data)
                
                # Log Artifacts
                if os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path)
                
                # End Run explicitly
                mlflow.end_run()
            
        except Exception as e:
            print(f"Error during final evaluation: {e}")
