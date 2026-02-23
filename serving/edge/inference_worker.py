import os
import queue
import threading
import time
from datetime import datetime
from uuid import uuid4

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

import config

# Load environment variables
load_dotenv()


class InferenceWorker(threading.Thread):
    """Consumes crops, runs inference, and pushes upload items."""

    def __init__(
        self,
        crop_queue: queue.Queue,
        upload_queue: queue.Queue,
        model_path: str,
        session_id: int = None,
        metrics=None,
        jpeg_quality: int | None = None,
        resize_max_width: int | None = None,
        resize_max_height: int | None = None,
        grayscale: bool | None = None,
    ):
        super().__init__(daemon=True)
        self.crop_queue = crop_queue
        self.upload_queue = upload_queue
        self.model_path = model_path
        self.session_id = session_id
        self.metrics = metrics
        self.running = False

        self.jpeg_quality = max(
            1,
            min(
                100,
                int(
                    jpeg_quality
                    if jpeg_quality is not None
                    else config.UPLOAD_IMAGE_JPEG_QUALITY
                ),
            ),
        )
        self.resize_max_width = int(
            resize_max_width
            if resize_max_width is not None
            else config.UPLOAD_IMAGE_MAX_WIDTH
        )
        self.resize_max_height = int(
            resize_max_height
            if resize_max_height is not None
            else config.UPLOAD_IMAGE_MAX_HEIGHT
        )
        self.use_grayscale = bool(
            grayscale if grayscale is not None else config.UPLOAD_IMAGE_GRAYSCALE
        )

        self.model = self._load_model(self.model_path)
        print(f"[InferenceWorker] 모델 로드 완료: {self.model_path}")
        print(
            "[InferenceWorker] upload image config: "
            f"jpeg_quality={self.jpeg_quality}, "
            f"max_size={self.resize_max_width}x{self.resize_max_height}, "
            f"grayscale={'on' if self.use_grayscale else 'off'}"
        )

    def _load_model(self, model_path: str):
        if model_path.endswith(".pt"):
            engine_path = model_path.replace(".pt", ".engine")
            if os.path.exists(engine_path):
                print(f"[InferenceWorker] TensorRT 엔진 발견: {engine_path}")
                return YOLO(engine_path, task="detect")

            print("[InferenceWorker] 엔진 파일 없음. FP16 TensorRT export 시도...")
            model = YOLO(model_path)
            try:
                model.export(format="engine", dynamic=True, device=0, half=True)
                if os.path.exists(engine_path):
                    return YOLO(engine_path, task="detect")
                return model
            except Exception as e:
                print(f"[InferenceWorker] 엔진 변환 실패, PT 모델로 진행: {e}")
                return model
        return YOLO(model_path, task="detect")

    def run(self):
        self.running = True
        print("[InferenceWorker] 추론 큐 모니터링 시작...")

        while self.running:
            try:
                item = self.crop_queue.get(timeout=1.0)
                if self.metrics:
                    self.metrics.update_queue_depth("crop_queue", self.crop_queue.qsize())

                camera_id = "unknown"
                crop = None
                frame_ts = None
                preprocess_done_ts = None

                if isinstance(item, dict):
                    camera_id = item.get("camera_id", "unknown")
                    crop = item.get("crop")
                    frame_ts = item.get("frame_ts")
                    preprocess_done_ts = item.get("preprocess_done_ts")
                elif isinstance(item, tuple) and len(item) >= 2:
                    camera_id = item[0]
                    crop = item[1]
                else:
                    self.crop_queue.task_done()
                    continue

                if crop is None:
                    self.crop_queue.task_done()
                    continue

                if preprocess_done_ts and self.metrics:
                    self.metrics.record_latency(
                        "crop_queue_wait_ms",
                        (time.time() - preprocess_done_ts) * 1000.0,
                    )

                start = time.time()
                results = self.model.predict(crop, conf=0.25, verbose=False)
                inference_ms = (time.time() - start) * 1000.0
                print(f"[InferenceWorker][{camera_id}] 추론 시간: {inference_ms:.1f}ms")
                if self.metrics:
                    self.metrics.record_inference(camera_id, inference_ms)

                payload, image_bytes = self._create_payload(camera_id, crop, results[0])
                upload_item = {
                    "payload": payload,
                    "image_bytes": image_bytes,
                    "meta": {
                        "camera_id": camera_id,
                        "frame_ts": frame_ts,
                        "inference_done_ts": time.time(),
                    },
                }

                if self.metrics:
                    self.metrics.record_upload_payload_bytes(len(image_bytes))

                try:
                    self.upload_queue.put_nowait(upload_item)
                    if self.metrics:
                        self.metrics.update_queue_depth(
                            "upload_queue", self.upload_queue.qsize()
                        )
                except queue.Full:
                    try:
                        self.upload_queue.get_nowait()
                        if self.metrics:
                            self.metrics.record_queue_drop("upload_queue")
                        self.upload_queue.put_nowait(upload_item)
                        if self.metrics:
                            self.metrics.update_queue_depth(
                                "upload_queue", self.upload_queue.qsize()
                            )
                    except queue.Empty:
                        pass

                self.crop_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[InferenceWorker] 루프 중 오류 발생: {e}")
                time.sleep(0.1)

    def _create_payload(self, camera_id, crop, result):
        qa_labels = {
            0: "Missing Hole",
            1: "Mouse Bite",
            2: "Open Circuit",
            3: "Short",
            4: "Spur",
            5: "Spurious Copper",
        }

        encoded = self._encode_upload_image(crop)
        if encoded is None:
            raise RuntimeError("JPEG encoding failed for upload payload")
        image_bytes, image_w, image_h, scale_x, scale_y = encoded

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            defect_name = qa_labels.get(cls_id, result.names.get(cls_id, f"class{cls_id}"))
            raw_bbox = [float(x) for x in box.xyxy[0].tolist()]
            scaled_bbox = self._scale_bbox(raw_bbox, scale_x, scale_y, image_w, image_h)
            detections.append(
                {
                    "defect_type": defect_name,
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": scaled_bbox,
                }
            )

        timestamp_now = datetime.now()
        image_id = (
            f"PCB_{camera_id}_"
            f"{timestamp_now.strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid4().hex[:8]}"
        )

        payload = {
            "timestamp": timestamp_now.isoformat(),
            "image_id": image_id,
            "camera_id": camera_id,
            "detections": detections,
            "session_id": self.session_id,
        }
        return payload, image_bytes

    def _scale_bbox(self, bbox, scale_x, scale_y, image_w, image_h):
        x1, y1, x2, y2 = bbox

        sx1 = int(round(x1 * scale_x))
        sy1 = int(round(y1 * scale_y))
        sx2 = int(round(x2 * scale_x))
        sy2 = int(round(y2 * scale_y))

        if image_w > 0:
            sx1 = max(0, min(image_w - 1, sx1))
            sx2 = max(0, min(image_w - 1, sx2))
        if image_h > 0:
            sy1 = max(0, min(image_h - 1, sy1))
            sy2 = max(0, min(image_h - 1, sy2))

        if image_w > 1 and sx2 <= sx1:
            sx2 = min(image_w - 1, sx1 + 1)
        if image_h > 1 and sy2 <= sy1:
            sy2 = min(image_h - 1, sy1 + 1)

        return [sx1, sy1, sx2, sy2]

    def _encode_upload_image(self, crop):
        image = crop

        if self.use_grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape[:2]
        max_w = self.resize_max_width if self.resize_max_width > 0 else w
        max_h = self.resize_max_height if self.resize_max_height > 0 else h
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        out_h, out_w = image.shape[:2]
        scale_x = out_w / float(w) if w > 0 else 1.0
        scale_y = out_h / float(h) if h > 0 else 1.0

        ok, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return None
        return buffer.tobytes(), out_w, out_h, scale_x, scale_y

    def stop(self):
        self.running = False
        print("[InferenceWorker] 중지 중...")
