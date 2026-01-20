"""
QAT (Quantization-Aware Training) 실행 스크립트

사용법:
    cd training
    uv run python run_qat.py --config config_qat.yaml

파이프라인:
    1. Pre-trained 모델 로드 (best.pt)
    2. Calibration (activation 범위 측정)
    3. QAT Fine-tuning
    4. ONNX Export (Q/DQ 노드 포함)
"""

import yaml
import argparse
import os
import importlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PCB Defect Detection QAT Runner")
    parser.add_argument('--config', type=str, default='config_qat.yaml',
                        help='Path to QAT config file')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip calibration step')
    args = parser.parse_args()

    # Config 파일 로드
    config_path = args.config
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_path = os.path.join(script_dir, config_path)
        if os.path.exists(candidate_path):
            print(f"Config '{config_path}' not found in CWD. Found at '{candidate_path}'.")
            config_path = candidate_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded QAT config from {args.config}")

    # 경로 해석 (상대 경로 -> 절대 경로)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # data_path 해석
    if 'data_path' in config and not os.path.isabs(config['data_path']):
        resolved_path = os.path.abspath(os.path.join(base_dir, config['data_path']))
        print(f"Resolving relative data_path '{config['data_path']}' -> '{resolved_path}'")
        config['data_path'] = resolved_path

    # qat.pretrained_path 해석
    qat_config = config.get('qat', {})
    pretrained_path = qat_config.get('pretrained_path', '')
    if pretrained_path and not os.path.isabs(pretrained_path):
        resolved_path = os.path.abspath(os.path.join(base_dir, pretrained_path))
        print(f"Resolving relative pretrained_path '{pretrained_path}' -> '{resolved_path}'")
        config['qat']['pretrained_path'] = resolved_path

    # Random Seed 설정
    from src.utils import set_seed, setup_logging, increment_path
    set_seed(config.get('seed', 42))
    print(f"Random seed set to: {config.get('seed', 42)}")

    # Ultralytics 설정
    from ultralytics import settings as ul_settings
    weights_dir = os.path.abspath(os.path.join(base_dir, "pretrained_weights"))
    ul_settings.update({'weights_dir': weights_dir})
    print(f"Ultralytics weights_dir set to: {weights_dir}")

    # 데이터셋 로드
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

    # 데이터 준비
    data_yaml = dataset.prepare()

    # 모델 로드 (QAT용)
    md_mod_name = config.get('model_module', 'yolov8s_qat')
    print(f"Loading Model Module: src/models/{md_mod_name}.py")

    try:
        md_lib = importlib.import_module(f"src.models.{md_mod_name}")
        model = md_lib.get_model(config)
    except ImportError as e:
        print(f"Error loading model module: {e}")
        return
    except AttributeError:
        print(f"Module src.models.{md_mod_name} must have a 'get_model(config)' function.")
        return

    # 저장 디렉토리 설정
    runs_dir = os.path.join(base_dir, "runs", "qat")
    config['project'] = runs_dir
    print(f"Set save project directory to: {runs_dir}")

    # 디렉토리 생성 및 로깅 설정
    initial_exp_name = config['exp_name']
    base_save_dir = os.path.join(runs_dir, initial_exp_name)
    save_dir = increment_path(Path(base_save_dir), exist_ok=False, mkdir=True)
    save_dir = str(save_dir)

    actual_exp_name = os.path.basename(save_dir)
    config['exp_name'] = actual_exp_name

    setup_logging(save_dir)

    # QAT Trainer 생성
    print(f"\n{'='*20} Start QAT Training (Exp: {config['exp_name']}) {'='*20}")
    print(f"Using data: {data_yaml}")

    from src.train_qat import QATTrainer
    trainer = QATTrainer(model, config)

    try:
        if args.skip_calibration:
            print("[QAT] Calibration 건너뜀 (--skip-calibration)")
            actual_save_dir = trainer.train(data_yaml)
        else:
            # 전체 파이프라인 실행 (calibration + training + export)
            actual_save_dir = trainer.run_full_pipeline(data_yaml)

        if actual_save_dir:
            save_dir = str(actual_save_dir)

        print("\nQAT Training completed.")

        # 최종 추론 (선택)
        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        onnx_path = os.path.join(save_dir, "weights", "best_qat.onnx")

        print(f"\n[결과 파일]")
        print(f"  - Best model: {best_model_path}")
        if os.path.exists(onnx_path):
            print(f"  - ONNX model: {onnx_path}")

        # 검증 실행
        if os.path.exists(best_model_path):
            _run_final_validation(best_model_path, data_yaml)

    except KeyboardInterrupt:
        print("\nQAT training interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during QAT training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        from src.utils import cleanup_artifacts
        cleanup_artifacts(save_dir, config)


def _run_final_validation(model_path: str, data_yaml: str) -> None:
    """최종 모델 검증"""
    print(f"\n{'='*20} Final Validation {'='*20}")

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        metrics = model.val(data=data_yaml, split='val', verbose=False)

        print(f"\n[QAT Final Results]")
        print(f"  - mAP50: {metrics.box.map50:.4f}")
        print(f"  - mAP50-95: {metrics.box.map:.4f}")

        # 클래스별 mAP
        if hasattr(metrics, 'ap_class_index'):
            print(f"\n{'Class':<20} | {'mAP50':<10}")
            print("-" * 35)
            for i, cls_idx in enumerate(metrics.ap_class_index):
                cls_idx = int(cls_idx)
                name = metrics.names.get(cls_idx, str(cls_idx))
                try:
                    res = metrics.class_result(i)
                    map50 = res[2]
                    print(f"{name:<20} | {map50:.4f}")
                except Exception:
                    pass
            print("-" * 35)

    except Exception as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    main()
