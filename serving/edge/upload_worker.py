import threading
import queue
import requests
import json
import os
import time
from datetime import datetime
import config

class UploadWorker(threading.Thread):
    """
    백엔드 서버로 추론 결과를 비동기 업로드하는 워커
    전송 실패 시 로컬 저장을 통해 데이터 손실 방지 (Failover)
    """

    def __init__(self, upload_queue: queue.Queue):
        super().__init__(daemon=True)
        self.upload_queue = upload_queue
        self.running = False
        
        # 저장 경로 생성
        os.makedirs(config.FAILED_DIR, exist_ok=True)

    def run(self):
        self.running = True
        print(f"[UploadWorker] 업로드 큐 모니터링 시작...")
        
        while self.running:
            try:
                # 큐에서 페이로드 가져오기
                payload = self.upload_queue.get(timeout=1.0)
                image_id = payload.get("image_id", "unknown")
                
                # 백엔드 전송 시도
                try:
                    # API Key 헤더 추가 (환경 변수에서 로드)
                    api_key = os.getenv("EDGE_API_KEY")
                    headers = {}
                    if api_key:
                        headers["X-API-KEY"] = api_key

                    response = requests.post(config.API_URL, json=payload, headers=headers, timeout=5.0)
                    if response.status_code in [200, 201]:
                        print(f"[UploadWorker] {image_id} 전송 성공!")
                    else:
                        print(f"[UploadWorker] {image_id} 전송 실패 (HTTP {response.status_code}): {response.text}")
                        self._save_locally(payload, image_id)
                except requests.exceptions.RequestException as e:
                    print(f"[UploadWorker] {image_id} 서버 연결 실패: {e}")
                    self._save_locally(payload, image_id)
                
                self.upload_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[UploadWorker] 루프 오류: {e}")

    def _save_locally(self, payload, image_id):
        """전송 실패한 데이터를 로컬에 JSON으로 저장"""
        filename = f"{image_id}_{datetime.now().strftime('%H%M%S')}.json"
        filepath = os.path.join(config.FAILED_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[UploadWorker] 데이터 로컬 저장 완료: {filepath}")
        except Exception as e:
            print(f"[UploadWorker] 로컬 저장 실패: {e}")

    def stop(self):
        self.running = False
        print("[UploadWorker] 중지 중...")
