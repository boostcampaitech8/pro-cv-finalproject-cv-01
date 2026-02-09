"""
YOLOv8s 모델 로더

config.yaml에서 model_module: "yolov8s"로 설정하면 이 모듈이 로드됩니다.
"""

from ultralytics import YOLO
import os


def get_model(config):
    """
    YOLOv8s 모델을 로드하여 반환합니다.

    Args:
        config: config.yaml에서 로드된 설정 딕셔너리

    Returns:
        YOLO: 로드된 YOLOv8s 모델
    """
    model_name = "yolov8s.pt"

    # pretrained_weights 디렉토리에서 먼저 찾기
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_path = os.path.join(base_dir, "pretrained_weights", model_name)

    if os.path.exists(weights_path):
        print(f"Loading model from: {weights_path}")
        model = YOLO(weights_path)
    else:
        # 없으면 ultralytics가 자동 다운로드
        print(f"Pretrained weights not found at {weights_path}")
        print(f"Downloading {model_name} from ultralytics...")
        model = YOLO(model_name)

    return model
