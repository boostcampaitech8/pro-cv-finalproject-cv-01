from ultralytics import YOLO
import os

def get_model(config):
    """
    Returns a YOLO model instance.
    The specific weight file is determined internally or by config.
    Default: yolo11x.pt
    """
    # Define default model for this module
    model_name = "yolo11x.pt"
    
    # 1. Check direct path
    if os.path.exists(model_name):
        pass # use as is
    # 2. Check pretrained_weights folder
    elif os.path.exists(os.path.join("pretrained_weights", model_name)):
        model_name = os.path.join("pretrained_weights", model_name)
    
    # 3. Initialize
    model = YOLO(model_name)
    return model
