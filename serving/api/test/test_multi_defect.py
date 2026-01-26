import requests
import time
import base64
import json
from datetime import datetime
import numpy as np
import cv2
from pathlib import Path

# 서버 URL
API_URL = "http://localhost:8001/detect/"

def create_dummy_image():
    # 검은색 배경에 무작위 사각형을 그린 더미 이미지 생성 (640x360)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "MULTI-DEFECT TEST", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode("utf-8")

def send_multi_defect_data():
    timestamp = datetime.now().isoformat()
    image_id = f"multi_test_{int(time.time()*1000)}"
    
    # 01_missing_hole_15.jpg 예시 참고 (3개의 결함)
    defect_names = {0: "missing_hole", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: "spur", 5: "spurious_copper"}
    
    detections = [
        {
            "confidence": 0.8117,
            "defect_type": defect_names[0],
            "bbox": [2460, 1111, 2531, 1151]
        },
        {
            "confidence": 0.8105,
            "defect_type": defect_names[0],
            "bbox": [2068, 1042, 2140, 1082]
        },
        {
            "confidence": 0.7705,
            "defect_type": defect_names[0],
            "bbox": [2072, 524, 2142, 565]
        }
    ]

    # 페이로드 구성
    payload = {
        "timestamp": timestamp,
        "image_id": image_id,
        "image": create_dummy_image(),
        "detections": detections
    }

    print(f"\n--- Sending Multi-Defect Data (Count: {len(detections)}) ---")
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            print(f"[Success] Response ID: {response.json().get('id')}")
        else:
            print(f"[Failed] Status Code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    send_multi_defect_data()
