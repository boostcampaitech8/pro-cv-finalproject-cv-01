import threading
import queue
import os
import base64
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
import config

# Load environment variables
load_dotenv()

class InferenceWorker(threading.Thread):
    """
    PCB 결함 탐지 추론 워커 스레드

    1. 모델 로드 (PT -> Engine 자동 변환)
    2. crop_queue에서 PCB 이미지를 가져와 추론
    3. 결과를 upload_queue로 전달
    """

    def __init__(self, crop_queue: queue.Queue, upload_queue: queue.Queue, model_path: str, session_id: int = None):
        """
        Args:
            crop_queue: 전처리기로부터 크롭된 이미지를 받는 큐
            upload_queue: 결과를 전송할 업로드 워커용 큐
            model_path: YOLO 모델 경로 (.pt 또는 .engine)
            session_id: 세션 ID (optional)
        """
        super().__init__(daemon=True)
        self.crop_queue = crop_queue
        self.upload_queue = upload_queue
        self.model_path = model_path
        self.session_id = session_id
        self.running = False

        # Hot-Swap Race Condition 방지용 Lock
        # reload_model()이 self.model을 교체하는 동안 predict()가
        # 동시에 실행되지 않도록 보호합니다.
        # Lock 범위는 predict() 호출 구간으로만 한정하여 지연을 최소화합니다.
        self._model_lock = threading.Lock()

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

    def reload_model(self):
        """
        런타임 중 새로운 엔진이 준비되었을 때 모델 객체를 안전하게 교체합니다.

        동작 순서:
          1. Lock 없이 새 모델을 먼저 로드 (시간이 걸리는 작업, 이 동안 추론 계속)
          2. Lock 획득 후 self.model 교체 (수μs 수준의 짧은 작업)
          3. 기존 모델 제거 및 Lock 해제
        """
        print(f"[InferenceWorker] 🔄 모델 핫스왑 시작...")
        try:
            # Lock 밖에서 먼저 로드 → 로드 중에도 추론 계속 가능
            new_model = self._load_model(self.model_path)
            # Lock 안에서만 포인터 교체 → predict()와 충돌 방지
            with self._model_lock:
                old_model = self.model
                self.model = new_model
                del old_model
            print(f"[InferenceWorker] ✅ 모델 Hot-Swap 완료!")
        except Exception as e:
            print(f"[InferenceWorker] ❌ 모델 리로드 실패. 기존 모델 유지: {e}")

    def run(self):
        self.running = True
        print(f"[InferenceWorker] 추론 큐 모니터링 시작...")
        
        while self.running:
            try:
                # 핫스왑 플래그 검사 (너무 자주 검사하지 않게 큐 대기 시간 이용)
                if os.path.exists(config.RELOAD_FLAG_PATH):
                    try:
                        self.reload_model()
                        os.remove(config.RELOAD_FLAG_PATH)
                    except OSError:
                        pass # 파일 지우기 충돌 무시

                # 큐에서 이미지 가져오기
                camera_id, crop = self.crop_queue.get(timeout=1.0)
                
                # 추론 수행 — Lock 안에서 실행하여 Hot-Swap과의 Race Condition 방지
                # reload_model()이 새 모델 포인터를 교체하는 순간과 충돌하지 않음
                # predict() 소요시간(~10-50ms)이 Lock 점유 시간이므로 영향 미미
                start = time.time()
                with self._model_lock:
                    results = self.model.predict(crop, conf=0.25, verbose=False)
                inference_time = time.time() - start
                print(f"[InferenceWorker][{camera_id}] 추론 시간: {inference_time*1000:.1f}ms")
                
                # 결과 포매팅
                payload = self._create_payload(camera_id, crop, results[0])

                
                # 업로드 큐에 추가
                try:
                    self.upload_queue.put_nowait(payload)
                except queue.Full:
                    # 업로드 큐가 가득 차면 가장 오래된 것 버림
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
                time.sleep(0.1)

    def _create_payload(self, camera_id, crop, result):
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
            cls_id = int(box.cls[0])
            defect_name = QA_LABELS.get(cls_id, result.names.get(cls_id, f"class{cls_id}"))
            
            detections.append({
                "defect_type": defect_name,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [int(float(x)) for x in box.xyxy[0].tolist()]
            })
 
        # 이미지 base64 인코딩
        _, buffer = cv2.imencode('.jpg', crop)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
 
        # image_id 생성
        timestamp_now = datetime.now()
        image_id = f"PCB_{camera_id}_{timestamp_now.strftime('%Y%m%d_%H%M%S')}"
 
        return {
            "timestamp": timestamp_now.isoformat(),
            "image_id": image_id,
            "camera_id": camera_id,
            "image": img_base64,
            "detections": detections,
            "session_id": self.session_id
        }

    def stop(self):
        """워커 종료"""
        self.running = False
        print("[InferenceWorker] 중지 중...")
