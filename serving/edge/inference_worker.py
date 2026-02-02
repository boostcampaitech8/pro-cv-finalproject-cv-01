import threading
import queue
import os
import base64
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InferenceWorker(threading.Thread):
    """
    PCB 결함 탐지 추론 워커 스레드

    1. 모델 로드 (PT -> Engine 자동 변환)
    2. crop_queue에서 PCB 이미지를 가져와 추론
    3. 결과를 upload_queue로 전달
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
        self.model = self._load_model(config.MODEL_PATH)
        print(f"[InferenceWorker] 모델 로드 완료: {config.MODEL_PATH}")

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
                print(f"[InferenceWorker] 엔진 파일이 없습니다. 변환을 시작합니다 (FP16 최적화)...")
                model = YOLO(model_path)
                try:
                    # Jetson Orin 최적화 (FP16, TensorRT)
                    model.export(format='engine', dynamic=True, device=0, half=True)
                    if os.path.exists(engine_path):
                        return YOLO(engine_path, task='detect')
                    return model
                except Exception as e:
                    print(f"[InferenceWorker] 엔진 변환 실패, 기본 모델로 진행합니다: {e}")
                    return model
        return YOLO(model_path, task='detect')

    def run(self):
        self.running = True
        print(f"[InferenceWorker] 추론 큐 모니터링 시작...")
        
        while self.running:
            try:
                # 큐에서 이미지 가져오기
                crop = self.crop_queue.get(timeout=1.0)
                
                # 추론 수행
                results = self.model.predict(crop, conf=0.25, verbose=False)
                
                # 결과 포매팅
                payload = self._create_payload(crop, results[0])
                
                # 업로드 큐에 추가
                try:
                    self.upload_queue.put_nowait(payload)
                except queue.Full:
                    # 업로드 큐가 가득 차면 가장 오래된 것 버리고 추가
                    try:
                        self.upload_queue.get_nowait()
                        self.upload_queue.put_nowait(payload)
                    except queue.Empty:
                        pass
                
                self.crop_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[InferenceWorker] 루프 중 오류 발생: {e}")

    def _create_payload(self, crop, result):
        """추론 결과를 백엔드 스펙에 맞는 페이로드로 구성"""
        
        # [NEW] 강제 매핑 테이블 (QAT 엔진 메타데이터 유실 대응)
        QA_LABELS = {
            0: "Missing Hole",
            1: "Mouse Bite",
            2: "Open Circuit",
            3: "Short",
            4: "Spur",
            5: "Spurious Copper"
        }
        
        detections = []
        for box in result.boxes:
            # 클래스 ID 확인
            cls_id = int(box.cls[0])
            # 매핑 시도 -> 없으면 기존 names -> 없으면 classN
            defect_name = QA_LABELS.get(cls_id, result.names.get(cls_id, f"class{cls_id}"))
            
            detections.append({
                "defect_type": defect_name,  # 수정된 이름 사용
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [int(float(x)) for x in box.xyxy[0].tolist()]
            })

        # 이미지 base64 인코딩
        _, buffer = cv2.imencode('.jpg', crop)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # image_id 생성
        timestamp_now = datetime.now()
        image_id = f"PCB_{timestamp_now.strftime('%Y%m%d_%H%M%S')}"

        return {
            "timestamp": timestamp_now.isoformat(),
            "image_id": image_id,
            "image": img_base64,
            "detections": detections,
            "session_id": self.session_id
        }

        # 백엔드 전송
        try:
            # API Key 헤더 추가 (환경 변수에서 로드)
            api_key = os.getenv("EDGE_API_KEY")
            headers = {}
            if api_key:
                headers["X-API-KEY"] = api_key
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=5.0)
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
