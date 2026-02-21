import base64
import json
import os
import queue
import threading
import time
from datetime import datetime
from uuid import uuid4

import requests

import config


class UploadWorker(threading.Thread):
    """
    Sends detection payloads to backend.

    Live path can use multipart(binary). Failed uploads are persisted as JSON
    with base64 image so scavenger can retry without additional file handling.
    """

    def __init__(self, upload_queue: queue.Queue, metrics=None, api_url: str = None):
        super().__init__(daemon=True)
        self.upload_queue = upload_queue
        self.metrics = metrics
        self.api_url = api_url or config.API_URL
        self.running = False
        os.makedirs(config.FAILED_DIR, exist_ok=True)

    def run(self):
        self.running = True
        print("[UploadWorker] 업로드 큐 모니터링 시작...")

        while self.running:
            try:
                item = self.upload_queue.get(timeout=1.0)
                if self.metrics:
                    self.metrics.update_queue_depth("upload_queue", self.upload_queue.qsize())

                payload = item
                image_bytes = None
                meta = {}

                if isinstance(item, dict) and "payload" in item:
                    payload = item.get("payload", {})
                    image_bytes = item.get("image_bytes")
                    meta = item.get("meta", {})

                if not isinstance(payload, dict):
                    self.upload_queue.task_done()
                    continue

                image_id = payload.get("image_id", "unknown")
                camera_id = payload.get("camera_id", meta.get("camera_id", "unknown"))

                if self.metrics and meta.get("inference_done_ts"):
                    self.metrics.record_latency(
                        "upload_queue_wait_ms",
                        (time.time() - meta["inference_done_ts"]) * 1000.0,
                    )

                start = time.time()
                ok, status_code, error_text = self._post_payload(payload, image_bytes)
                upload_ms = (time.time() - start) * 1000.0
                if self.metrics:
                    self.metrics.record_latency("upload_ms", upload_ms)
                    self.metrics.record_upload_result(ok, source="live")

                if ok:
                    print(f"[UploadWorker][{camera_id}] {image_id} 전송 성공!")
                    frame_ts = meta.get("frame_ts")
                    if frame_ts and self.metrics:
                        self.metrics.record_latency("e2e_ms", (time.time() - frame_ts) * 1000.0)
                else:
                    if status_code is not None:
                        print(
                            f"[UploadWorker][{camera_id}] {image_id} 전송 실패 "
                            f"(HTTP {status_code}): {error_text}"
                        )
                    else:
                        print(f"[UploadWorker][{camera_id}] {image_id} 서버 연결 실패: {error_text}")
                    self._save_locally(payload, image_id, image_bytes=image_bytes)

                self.upload_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[UploadWorker] 루프 오류: {e}")

    def _post_payload(self, payload: dict, image_bytes: bytes | None):
        transport = (config.UPLOAD_TRANSPORT or "multipart").strip().lower()
        if transport == "json":
            json_payload = self._json_payload(payload, image_bytes=image_bytes)
            return self._post_json(json_payload)

        ok, status_code, error_text = self._post_multipart(payload, image_bytes=image_bytes)
        if ok:
            return True, status_code, error_text

        # Compatibility fallback for legacy backend that only accepts JSON base64.
        fallback_enabled = bool(getattr(config, "UPLOAD_JSON_FALLBACK_ENABLED", True))
        if fallback_enabled and status_code in (400, 404, 405, 415, 422):
            print(
                f"[UploadWorker] multipart rejected (HTTP {status_code}), "
                "fallback to JSON(base64)"
            )
            json_payload = self._json_payload(payload, image_bytes=image_bytes)
            return self._post_json(json_payload)

        return False, status_code, error_text

    def _build_headers(self):
        api_key = os.getenv("EDGE_API_KEY")
        return {"X-API-KEY": api_key} if api_key else {}

    def _post_multipart(self, payload: dict, image_bytes: bytes | None):
        headers = self._build_headers()
        image_id = payload.get("image_id", "image")

        data = {
            "timestamp": payload.get("timestamp", ""),
            "image_id": image_id,
            "detections": json.dumps(payload.get("detections", []), ensure_ascii=False),
        }

        camera_id = payload.get("camera_id")
        if camera_id:
            data["camera_id"] = camera_id

        session_id = payload.get("session_id")
        if session_id is not None:
            data["session_id"] = str(session_id)

        files = None
        if image_bytes:
            files = {"image_file": (f"{image_id}.jpg", image_bytes, "image/jpeg")}

        try:
            response = requests.post(
                self.api_url,
                data=data,
                files=files,
                headers=headers,
                timeout=5.0,
            )
            if response.status_code in (200, 201):
                return True, response.status_code, ""
            return False, response.status_code, response.text
        except requests.exceptions.RequestException as e:
            return False, None, str(e)

    def _post_json(self, payload: dict):
        headers = self._build_headers()
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=5.0)
            if response.status_code in (200, 201):
                return True, response.status_code, ""
            return False, response.status_code, response.text
        except requests.exceptions.RequestException as e:
            return False, None, str(e)

    def _json_payload(self, payload: dict, image_bytes: bytes | None):
        payload_json = dict(payload)
        if image_bytes is not None and "image" not in payload_json:
            payload_json["image"] = base64.b64encode(image_bytes).decode("utf-8")
        return payload_json

    def _save_locally(self, payload, image_id, image_bytes: bytes | None = None):
        safe_image_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in image_id)
        filename = (
            f"{safe_image_id}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid4().hex[:8]}.json"
        )
        filepath = os.path.join(config.FAILED_DIR, filename)

        payload_to_save = self._json_payload(payload, image_bytes=image_bytes)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload_to_save, f, ensure_ascii=False, indent=2)
            print(f"[UploadWorker] 데이터 로컬 저장 완료: {filepath}")
        except Exception as e:
            print(f"[UploadWorker] 로컬 저장 실패: {e}")

    def stop(self):
        self.running = False
        print("[UploadWorker] 중지 중...")
