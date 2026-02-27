class PCBTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, data_yaml):
        """
        특정 데이터(data.yaml) 파일에 대한 학습을 실행합니다.
        """
        import wandb

        
        # MLflow 설정
        # 1. Ultralytics의 자동 MLflow 연동 기능 비활성화 (충돌 방지)
        from ultralytics import settings as ul_settings
        ul_settings.update({'mlflow': False})
        
        # 2. 로컬 MLflow 경로 및 실험명 설정
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("PCB_Res_Experiments")
        
        # WandB 설정
        if self.config.get('wandb_project'):
            wandb.init(project=self.config['wandb_project'], name=f"{self.config['exp_name']}_fold", reinit=True)
            
        # 스케줄러(Scheduler) 설정
        # YOLOv8은 'cos_lr' 인자를 사용합니다: True=Cosine, False=Linear
        cos_lr = True if self.config.get('scheduler_type') == 'cosine' else False
        
        # 옵티마이저(Optimizer) 설정
        # 참고: YOLOv8 내부에서 옵티마이저 생성을 처리하므로 이름과 파라미터만 전달합니다.
        
        # 사용자 정의 프로그레스 바(Progress Bar)를 위한 콜백 추가
        # 참고: 진행률 표시를 위해 기본 YOLO 로깅 방식을 활용합니다.
        from ultralytics.utils import LOGGER
        import logging
        import time
        
        # Custom Progress Bar Logic
        import sys

        # 일반적인 YOLO 정보성 로그 숨김 (데이터셋 스캔 등)
        LOGGER.setLevel(logging.WARNING)

        # 기본 Ultralytics MLflow 콜백 비활성화 (Run 'not found' 에러 방지)
        # mlflow 패키지가 설치되어 있으면 Ultralytics가 자동 로깅을 시도하므로 이를 수동으로 해제합니다.
        from ultralytics.utils.callbacks import mlflow as mlflow_callbacks
        
        # 등록된 MLflow 콜백이 있다면 안전하게 제거합니다.
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
        except ImportError:
            pass
        
        epoch_start_time = 0
        current_batch_idx = 0

        def on_train_epoch_start(trainer):
            nonlocal epoch_start_time, current_batch_idx
            epoch_start_time = time.time()
            current_batch_idx = 0 # 배치 카운터 초기화

        def on_train_batch_end(trainer):
            """수동 프로그레스 바: \\r을 사용하여 같은 줄에 진행 상태를 덮어씁니다."""
            nonlocal current_batch_idx
            current_batch_idx += 1
            
            current_epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            current_batch = current_batch_idx
            # 전체 배치 수를 가져옵니다.
            total_batches = len(trainer.train_loader)
            
            # Loss 항목들 (box, cls, dfl) - 보통 (3,) 형태의 텐서입니다.
            # 존재하는지 확인 후 값을 추출합니다.
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                # Loss 값을 가독성 좋게 포맷팅합니다.
                losses = [f"{x.item():.4f}" for x in trainer.loss_items]
                loss_str = f"box: {losses[0]} cls: {losses[1]} dfl: {losses[2]}"
            else:
                loss_str = "loss: N/A"

            # 프로그레스 바 UI 형태 구성
            percent = int((current_batch / total_batches) * 100)
            bar_len = 20
            filled_len = int(bar_len * current_batch // total_batches)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            
            # \r (Carriage Return)을 사용하여 콘솔 출력
            # 예시: Epoch 1/200 [████------] 40% | box: 1.2 cls: 0.5 ...
            sys.stdout.write(f"\rEpoch {current_epoch}/{total_epochs} [{bar}] {percent}% | {loss_str}")
            sys.stdout.flush()

        def on_fit_epoch_end(trainer):
            """에포크 종료 시 요약 정보 출력"""
            nonlocal epoch_start_time
            metrics = trainer.metrics
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            duration = time.time() - epoch_start_time
            
            # 프로그레스 바를 덮어쓰고 깔끔하게 최종 검증 결과를 요약하여 출력합니다.
            
            # mAP 지표 가져오기
            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
            
            # YOLO Fitness 계산: 0.1 * mAP50 + 0.9 * mAP50-95
            fitness = (0.1 * map50) + (0.9 * map50_95)
            
            # Loss 추출 (해당 에포크의 마지막 배치 기준)
            if hasattr(trainer, 'loss_items'):
                box_loss = trainer.loss_items[0].item()
                cls_loss = trainer.loss_items[1].item()
                dfl_loss = trainer.loss_items[2].item()
            else:
                box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0

            # 피트니스(Fitness)와 함께 출력
            print(f"Epoch {epoch}/{total_epochs} | box: {box_loss:.4f} cls: {cls_loss:.4f} dfl: {dfl_loss:.4f} | mAP50: {map50:.4f} | Fitness: {fitness:.4f} | Time: {duration:.2f}s")
            
            # MLflow 지표 로깅 (에포크 단위)
            if mlflow.active_run():
                mlflow.log_metrics({
                    "train/box_loss": box_loss,
                    "train/cls_loss": cls_loss,
                    "train/dfl_loss": dfl_loss,
                    "metrics/mAP50": map50,
                    "metrics/mAP50-95": map50_95,
                    "metrics/fitness": fitness
                }, step=epoch)
            
            # 클래스별 mAP50 지표가 있을 경우 출력
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

        # 콜백 등록
        self.model.add_callback("on_train_epoch_start", on_train_epoch_start)
        self.model.add_callback("on_train_batch_end", on_train_batch_end)
        self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # MLflow Run 시작
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
            name=self.config['exp_name'], # 사용자 정의 실험명 적용
            exist_ok=True, 
            
            # 옵티마이저 및 스케줄링 설정
            optimizer=self.config.get('optimizer', 'auto'),
            lr0=self.config.get('lr0', 0.01),
            lrf=self.config.get('lrf', 0.01),
            momentum=self.config.get('momentum', 0.937),
            weight_decay=self.config.get('weight_decay', 0.0005),
            cos_lr=cos_lr,
            # 데이터 증강 기법 (config.yaml을 통해 제어)
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
            
            # 시각화 기능 지원
            plots=True, # JPG 결과물 저장 활성화
            
            # Loss 가중치
            box=self.config.get('box', 7.5),
            cls=self.config.get('cls', 0.5),
            dfl=self.config.get('dfl', 1.5),
            
            verbose=False,  # 기본 TQDM 바 비활성화 (수동 프로그레스 바를 사용하므로)
            val=True       # 검증(Validation) 활성화
        )
            
        except Exception as e:
            print(f"학습 실패: {e}")
            if mlflow.active_run():
                mlflow.end_run()
            raise e
        
        # 최고 성능 모델(Best Model) 최종 평가 수행
        self.run_final_eval(data_yaml)
        
        # 파일 경로 자동 증가 처리(예: baseline2) 문제를 방지하기 위해 실제 저장 경로를 반환합니다.
        return self.model.trainer.save_dir

    def run_final_eval(self, data_yaml):
        """
        최저 Loss/최고 mAP를 기록한 최고 성능 모델(best.pt)을 로드하여 
        최종 검증 및 지표 요약을 출력합니다.
        """
        import os

        from ultralytics import YOLO
        import mlflow
        
        # trainer 객체 및 저장 경로 유효성 검사
        if not hasattr(self.model, 'trainer') or not self.model.trainer:
            print("Trainer가 초기화되지 않았습니다. 최종 평가를 건너뜁니다.")
            return

        save_dir = self.model.trainer.save_dir
        # YOLOv8의 기본 모델 저장 구조: weights/best.pt
        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        
        if not os.path.exists(best_model_path):
            print(f"경고: {best_model_path} 경로에서 최고 성능 모델을 찾을 수 없습니다. 최종 평가를 건너뜁니다.")
            return

        print(f"\n{'='*20} 최종 평가 (최고 성능 모델 기준) {'='*20}")
        print(f"최고 성능 모델 로드 경로: {best_model_path}")
        
        try:
            # 최고 성능 모델 로드
            best_model = YOLO(best_model_path)
            
            # 검증 수행
            # verbose=False로 설정하여 중복 테이블 출력을 방지합니다.
            print("모델 검증을 실행 중입니다...")
            metrics = best_model.val(data=data_yaml, split='val', verbose=False)
            
            print("\n[최고 성능 모델 클래스별 평가 지표]")
            print(f"{'Class':<20} | {'mAP50':<10} | {'mAP50-95':<10}")
            print("-" * 50)
            
            names = metrics.names
            if hasattr(metrics, 'ap_class_index'):
                for i, cls_idx in enumerate(metrics.ap_class_index):
                    cls_idx = int(cls_idx)
                    name = names.get(cls_idx, str(cls_idx))
                    
                    # Ultralytics 내부 구조에 의지하여 mAP 가져오기
                    try:
                        res = metrics.class_result(i)
                        map50 = res[2]
                        map50_95 = res[3]
                        print(f"{name:<20} | {map50:.4f}     | {map50_95:.4f}")
                    except Exception:
                        # 구조가 달라서 실패할 경우 대비 (Fallback)
                        print(f"{name:<20} | 불가능        | 불가능")
                        
            print("-" * 50)
            print(f"Overall mAP50: {metrics.box.map50:.4f}")
            print(f"Overall mAP50-95: {metrics.box.map:.4f}")
            print(f"{'='*50}\n")
            
            # 최종 지표를 MLflow에 로깅합니다.
            if mlflow.active_run():
                # 1. 전역(Global) 지표
                log_data = {
                    "final_mAP50": metrics.box.map50,
                    "final_mAP50-95": metrics.box.map
                }
                
                # 2. 클래스별 분류 지표 (Class-wise)
                try:
                    names = metrics.names
                    if hasattr(metrics, 'ap_class_index'):
                        for i, cls_idx in enumerate(metrics.ap_class_index):
                            cls_idx = int(cls_idx)
                            name = names.get(cls_idx, str(cls_idx))
                            
                            # 특수문자나 띄어쓰기가 MLflow 키 등록에 제한이 되지 않게 밑줄로 변환합니다.
                            safe_name = name.replace(" ", "_")
                            
                            res = metrics.class_result(i)
                            map50_cls = res[2]
                            map50_95_cls = res[3]
                            
                            log_data[f"final_mAP50_{safe_name}"] = map50_cls
                            log_data[f"final_mAP95_{safe_name}"] = map50_95_cls
                except Exception as e:
                    print(f"경고: 클래스 지표 로깅 실패: {e}")

                mlflow.log_metrics(log_data)
                
                # 모델 아티팩트 저장 (MLflow)
                if os.path.exists(best_model_path):
                    mlflow.log_artifact(best_model_path)
                
                # 트래킹 종료
                mlflow.end_run()
            
        except Exception as e:
            print(f"최종 모델 평가 중 에러 발생: {e}")
