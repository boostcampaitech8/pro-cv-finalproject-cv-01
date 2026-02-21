"""
Edge runtime metrics collector.

Collects per-camera counters, queue watermarks, and latency percentiles.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional


def _percentile(values: Deque[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = int(round((len(ordered) - 1) * (p / 100.0)))
    return ordered[index]


def _sum_map(data: Dict[str, int]) -> int:
    return sum(data.values())


class EdgeMetrics:
    """Thread-safe metrics storage used by receiver/main/inference/upload workers."""

    def __init__(self, latency_buffer_size: int = 20000):
        self._lock = threading.Lock()
        self.started_at = time.time()

        # Per-camera counters
        self.input_frames = defaultdict(int)
        self.input_drops = defaultdict(int)
        self.processed_frames = defaultdict(int)
        self.crops = defaultdict(int)
        self.inference_count = defaultdict(int)

        # Queue drop counters
        self.queue_drops = defaultdict(int)

        # Upload/retry counters
        self.upload_success = 0
        self.upload_fail = 0
        self.scavenger_success = 0
        self.scavenger_fail = 0
        self.scavenger_expired = 0

        # Queue high-water marks
        self.queue_high_watermark = defaultdict(int)
        self.queue_current_depth = defaultdict(int)

        # Latency buckets (ms)
        self.latencies = {
            "frame_queue_wait_ms": deque(maxlen=latency_buffer_size),
            "preprocess_ms": deque(maxlen=latency_buffer_size),
            "crop_queue_wait_ms": deque(maxlen=latency_buffer_size),
            "inference_ms": deque(maxlen=latency_buffer_size),
            "upload_queue_wait_ms": deque(maxlen=latency_buffer_size),
            "upload_ms": deque(maxlen=latency_buffer_size),
            "e2e_ms": deque(maxlen=latency_buffer_size),
            "upload_payload_kb": deque(maxlen=latency_buffer_size),
        }

    def record_input(self, camera_id: str):
        with self._lock:
            self.input_frames[camera_id] += 1

    def record_input_drop(self, camera_id: str):
        with self._lock:
            self.input_drops[camera_id] += 1

    def record_processed(self, camera_id: str):
        with self._lock:
            self.processed_frames[camera_id] += 1

    def record_crop(self, camera_id: str):
        with self._lock:
            self.crops[camera_id] += 1

    def record_inference(self, camera_id: str, latency_ms: float):
        with self._lock:
            self.inference_count[camera_id] += 1
            self.latencies["inference_ms"].append(latency_ms)

    def record_upload_result(self, success: bool, source: str = "live"):
        with self._lock:
            if source == "scavenger":
                if success:
                    self.scavenger_success += 1
                else:
                    self.scavenger_fail += 1
            else:
                if success:
                    self.upload_success += 1
                else:
                    self.upload_fail += 1

    def record_scavenger_expired(self):
        with self._lock:
            self.scavenger_expired += 1

    def record_queue_drop(self, queue_name: str):
        with self._lock:
            self.queue_drops[queue_name] += 1

    def record_latency(self, name: str, latency_ms: float):
        if name not in self.latencies:
            return
        with self._lock:
            self.latencies[name].append(latency_ms)

    def record_upload_payload_bytes(self, size_bytes: int):
        with self._lock:
            self.latencies["upload_payload_kb"].append(size_bytes / 1024.0)

    def update_queue_depth(self, queue_name: str, depth: int):
        with self._lock:
            self.queue_current_depth[queue_name] = depth
            if depth > self.queue_high_watermark[queue_name]:
                self.queue_high_watermark[queue_name] = depth

    def snapshot(self, failed_dir: Optional[str] = None) -> Dict:
        with self._lock:
            now = time.time()
            elapsed = max(now - self.started_at, 1e-6)

            latency_stats = {}
            for key, values in self.latencies.items():
                latency_stats[key] = {
                    "p50": _percentile(values, 50),
                    "p95": _percentile(values, 95),
                    "count": len(values),
                }

            upload_total = self.upload_success + self.upload_fail
            upload_success_rate = (self.upload_success / upload_total * 100.0) if upload_total > 0 else 0.0

            snap = {
                "ts": now,
                "elapsed_s": elapsed,
                "input_frames": dict(self.input_frames),
                "input_drops": dict(self.input_drops),
                "processed_frames": dict(self.processed_frames),
                "crops": dict(self.crops),
                "inference_count": dict(self.inference_count),
                "queue_drops": dict(self.queue_drops),
                "queue_current_depth": dict(self.queue_current_depth),
                "queue_high_watermark": dict(self.queue_high_watermark),
                "upload_success": self.upload_success,
                "upload_fail": self.upload_fail,
                "upload_success_rate": upload_success_rate,
                "scavenger_success": self.scavenger_success,
                "scavenger_fail": self.scavenger_fail,
                "scavenger_expired": self.scavenger_expired,
                "latency": latency_stats,
                "totals": {
                    "input_frames": _sum_map(self.input_frames),
                    "processed_frames": _sum_map(self.processed_frames),
                    "crops": _sum_map(self.crops),
                    "inference": _sum_map(self.inference_count),
                    "input_drops": _sum_map(self.input_drops),
                },
            }

        if failed_dir:
            try:
                backlog = len([name for name in os.listdir(failed_dir) if name.endswith(".json")])
            except FileNotFoundError:
                backlog = 0
            snap["failed_backlog"] = backlog
        else:
            snap["failed_backlog"] = 0

        return snap


def format_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"
