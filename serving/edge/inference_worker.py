import threading
import queue
import time
import requests
import os
import base64
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

class InferenceWorker(threading.Thread):
    """
    PCB 결함 탐지 추론 워커 스레드

    1. 모델 로드 (PT -> Engine 자동 변환)
    2. crop_queue에서 PCB 이미지를 가져와 추론
    3. 결과를 백엔드 API로 전송
    """

    def __init__(self, crop_queue: queue.Queue, model_path: str, api_url: str, session_id: int = None):
        """
        Args:
            crop_queue: 전처리기로부터 크롭된 이미지를 받는 큐
            model_path: YOLO 모델 경로 (.pt 또는 .engine)
            api_url: 결과를 전송할 백엔드 API 주소
            session_id: 세션 ID (optional)
        """
        super().__init__(daemon=True)
        self.crop_queue = crop_queue
        self.model_path = model_path
        self.api_url = api_url
        self.session_id = session_id
        self.running = False

        # 모델 로드 (Engine 변환 포함)
        self.model = self._load_model(model_path)
        print(f"[InferenceWorker] 모델 로드 완료: {model_path}")

    def _load_model(self, model_path: str):
        """
        모델을 로드합니다. .pt 파일인 경우 .engine 파일이 있는지 확인하고
        없으면 변환을 수행합니다.
        """
        if model_path.endswith('.pt'):
            engine_path = model_path.replace('.pt', '.engine')
            if os.path.exists(engine_path):
                print(f"[InferenceWorker] TensorRT 엔진 발견: {engine_path}")
                return YOLO(engine_path, task='detect')
            else:
                print(f"[InferenceWorker] 엔진 파일이 없습니다. 변환을 시작합니다 (시간이 걸릴 수 있음)...")
                model = YOLO(model_path)
                # Jetson(TensorRT) 최적화를 위해 export 실행
                # device=0 (GPU) 사용, dynamic=True (가변 크기 대응)
                try:
                    model.export(format='engine', dynamic=True, device=0, half=True)
                    return YOLO(engine_path, task='detect')
                except Exception as e:
                    print(f"[InferenceWorker] 엔진 변환 실패, 기본 모델로 진행합니다: {e}")
                    return model
        return YOLO(model_path, task='detect')

    def run(self):
        self.running = True
        print(f"[InferenceWorker] 추론 큐 모니터링 시작...")
        
        while self.running:
            try:
                # 큐에서 이미지 가져오기 (타임아웃 설정으로 종료 체크 가능하게 함)
                crop = self.crop_queue.get(timeout=1.0)
                
                # 추론 수행
                # verbose=False로 로그 최소화, conf 임계값 설정
                results = self.model.predict(crop, conf=0.25, verbose=False)
                
                # 결과 처리 및 전송
                self._handle_results(crop, results[0])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[InferenceWorker] 루프 중 오류 발생: {e}")
                time.sleep(1) # 오류 발생 시 잠시 대기

    def _handle_results(self, crop, result):
        """추론 결과를 파싱하고 백엔드로 전송"""
        detections = []
        for box in result.boxes:
            # bbox: [x1, y1, x2, y2] 
            # 백엔드 스키마(schemas.py) 형식에 맞춤
            detections.append({
                "defect_type": result.names[int(box.cls[0])], # class_name -> defect_type
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [int(float(x)) for x in box.xyxy[0].tolist()] # 정수형 리스트로 변환
            })

        # 이미지 base64 인코딩 (전송용)
        _, buffer = cv2.imencode('.jpg', crop)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # image_id 생성 (예: PCB_20260128_224601)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = f"PCB_{timestamp_str}"

        # 백엔드 DetectRequest 스펙에 맞춘 페이로드 구성
        payload = {
            "timestamp": datetime.now().isoformat(),
            "image_id": image_id,
            "image": img_base64,
            "detections": detections,
            "session_id": self.session_id
        }

        # 백엔드 전송
        try:
            response = requests.post(self.api_url, json=payload, timeout=5.0)
            if response.status_code == 200 or response.status_code == 201:
                print(f"[InferenceWorker] {image_id} 전송 성공! (탐지 개수: {len(detections)})")
            else:
                print(f"[InferenceWorker] 전송 실패: HTTP {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[InferenceWorker] API 서버 연결 실패: {e}")

    def stop(self):
        """워커 종료"""
        self.running = False
        print("[InferenceWorker] 중지 중...")
