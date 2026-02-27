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

    def __init__(self, crop_queue: queue.Queue, upload_queue: queue.Queue, model_path: str, session_id: int = None, session_url: str = None):
        """
        Args:
            crop_queue: 전처리기로부터 크롭된 이미지를 받는 큐
            upload_queue: 결과를 전송할 업로드 워커용 큐
            model_path: YOLO 모델 경로 (.pt 또는 .engine)
            session_id: 세션 ID (optional)
            session_url: 세션 API 주소 (핫스왑 시 새 세션 발급용)
        """
        super().__init__(daemon=True)
        self.crop_queue = crop_queue
        self.upload_queue = upload_queue
        self.model_path = model_path
        self.session_id = session_id
        self.session_url = session_url
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

    def _get_model_name(self) -> str:
        """현재 활성 모델 버전을 반환합니다.

        머신마다 동일한 로직이 반복되어 있어 메서드로 분리했습니다.
        """
        version_file = os.path.join(os.path.dirname(config.MODEL_PATH), "current_version.json")
        model_name = "yolov11m_v0"
        if os.path.exists(version_file):
            try:
                import json
                with open(version_file, "r") as f:
                    v_info = json.load(f)
                    model_name = v_info.get("model_name", model_name)
            except Exception:
                pass
        return model_name

    def reload_model(self):
        """
        런타임 중 새로운 엔진이 준비되었을 때 모델 객체를 안전하게 교체하고 세션을 갱신합니다.
        """
        print(f"[InferenceWorker] 🔄 모델 핫스왑 시작...")
        try:
            new_model = self._load_model(self.model_path)
            with self._model_lock:
                old_model = self.model
                self.model = new_model
                del old_model
                
            # 세션 갱신 (기존 세션 종료 -> 새 세션 시작)
            if self.session_url:
                import requests
                # 1. 기존 세션 종료
                if self.session_id is not None:
                    try:
                        requests.patch(f"{self.session_url}{self.session_id}", timeout=5.0)
                        print(f"[InferenceWorker] 기존 세션({self.session_id}) 종료 완료")
                    except Exception as e:
                        print(f"[InferenceWorker] 세션 종료 요청 실패: {e}")
                
                # 2. 새 모델 버전 읽어오기
                version_file = os.path.join(os.path.dirname(config.MODEL_PATH), "current_version.json")
                mlops_ver, yolo_ver = "v0", "yolov11m"
                if os.path.exists(version_file):
                    try:
                        import json
                        with open(version_file, "r") as f:
                            v_info = json.load(f)
                            mlops_ver = v_info.get("mlops_version", "v0")
                            yolo_ver = v_info.get("yolo_version", "yolov11m")
                    except Exception as e:
                        pass
                
                # 3. 새 세션 시작
                try:
                    payload = {"model_name": model_name}
                    response = requests.post(self.session_url, json=payload, timeout=5.0)
                    if response.status_code == 201:
                        self.session_id = response.json().get("id")
                        print(f"[InferenceWorker] 새 세션 시작 완료: ID={self.session_id} 모델명={model_name}")
                    else:
                        print(f"[InferenceWorker] 새 세션 시작 실패: HTTP {response.status_code}")
                except Exception as e:
                    print(f"[InferenceWorker] 세션 교체 요청 실패: {e}")

            print(f"[InferenceWorker] ✅ 모델 Hot-Swap 프로세스 완료!")
        except Exception as e:
            print(f"[InferenceWorker] ❌ 모델 리로드 실패. 기존 모델 유지: {e}")

    def run(self):
        self.running = True
        print(f"[InferenceWorker] 추론 큐 모니터링 시작...")
        # busy 플래그는 "실제로 추론을 수행하는 동안"만 유지합니다.
        busy = False
        last_active = 0.0
        idle_timeout = getattr(config, 'INFERENCE_IDLE_TIMEOUT_S', 5)

        try:
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

                    # 실제 추론을 시작하면 busy 플래그 생성
                    if not busy:
                        try:
                            open(config.INFERENCE_BUSY_FLAG_PATH, 'w').close()
                            busy = True
                            print("[InferenceWorker] 추론 활성화 — busy 플래그 생성")
                        except Exception:
                            pass

                    # 추론 수행 — Lock 안에서 실행하여 Hot-Swap과의 Race Condition 방지
                    start = time.time()
                    with self._model_lock:
                        results = self.model.predict(crop, conf=0.25, verbose=False)
                    inference_time = time.time() - start
                    last_active = time.time()
                    print(f"[InferenceWorker][{camera_id}] 추론 시간: {inference_time*1000:.1f}ms")

                    # 결과 포매팅
                    payload = self._create_payload(camera_id, crop, results[0])

                    # 업로드 큐에 추가
                    try:
                        self.upload_queue.put_nowait(payload)
                    except queue.Full:
                        try:
                            self.upload_queue.get_nowait()
                            self.upload_queue.put_nowait(payload)
                        except queue.Empty:
                            pass

                    self.crop_queue.task_done()

                except queue.Empty:
                    # 큐가 비어있고 일정 시간(idle_timeout) 경과하면 busy 플래그 제거
                    if busy and (time.time() - last_active) >= idle_timeout:
                        try:
                            if os.path.exists(config.INFERENCE_BUSY_FLAG_PATH):
                                os.remove(config.INFERENCE_BUSY_FLAG_PATH)
                            print("[InferenceWorker] 추론 비활성화 — busy 플래그 제거 (idle)")
                        except Exception:
                            pass
                        busy = False
                    continue
                except Exception as e:
                    print(f"[InferenceWorker] 루프 중 오류 발생: {e}")
                    time.sleep(0.1)
        finally:
            # 워커 종료 시 busy 플래그 제거
            try:
                if os.path.exists(config.INFERENCE_BUSY_FLAG_PATH):
                    os.remove(config.INFERENCE_BUSY_FLAG_PATH)
            except Exception:
                pass

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
 
        payload = {
            "timestamp": timestamp_now.isoformat(),
            "image_id": image_id,
            "camera_id": camera_id,
            "image": img_base64,
            "detections": detections,
            "session_id": self.session_id
        }
        # 모델명이 존재하면 함께 포함
        model_name = self._get_model_name()
        if model_name:
            payload["model_name"] = model_name
        return payload

    def stop(self):
        """워커 종료"""
        self.running = False
        print("[InferenceWorker] 중지 중...")
