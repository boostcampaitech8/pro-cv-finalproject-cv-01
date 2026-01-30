import requests
import time
import base64
import random
from datetime import datetime
import numpy as np
import cv2

# 서버 URL
API_URL = "http://localhost:8000/detect/"

def create_dummy_image():
    # 검은색 배경에 무작위 사각형을 그린 더미 이미지 생성 (640x360)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "DUMMY DEFECT DATA", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode("utf-8")

def send_dummy_data(is_defect=True):
    timestamp = datetime.now().isoformat()
    image_id = f"test_{int(time.time()*1000)}"
    
    detections = []
    if is_defect:
        # 결함이 1~2개 있는 경우
        num_defects = random.randint(1, 2)
        defect_types = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
        
        for _ in range(num_defects):
            detections.append({
                "confidence": round(random.uniform(0.7, 0.99), 4),
                "defect_type": random.choice(defect_types),
                "bbox": [random.randint(0, 100), random.randint(0, 100), random.randint(200, 300), random.randint(200, 300)]
            })

    # 페이로드 구성 (rtsp_to_server.py 규격 참고)
    payload = {
        "timestamp": timestamp,
        "image_id": image_id,
        "result": "defect" if detections else "normal",
        "confidence": detections[0]["confidence"] if detections else 0.0,
        "defect_type": detections[0]["defect_type"] if detections else None,
        "bbox": detections[0]["bbox"] if detections else None,
        "image": create_dummy_image() if detections else None,
        "detections": detections
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"[Success] Sent {payload['result']} data. ID: {image_id}")
        else:
            print(f"[Failed] Status Code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    print("Starting Dummy Data Injection Test...")
    # 총 5개의 데이터를 전송 (3개 결함, 2개 정상)
    test_cases = [True, False, True, False, True]
    
    for i, is_defect in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        send_dummy_data(is_defect)
        time.sleep(1) # 1초 간격

    print("\nTest Finished.")
