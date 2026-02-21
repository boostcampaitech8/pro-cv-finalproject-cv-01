"""
PCB 전처리 파이프라인 메인 모듈

RTSP 영상을 수신하고, PCB를 감지/크롭하여 추론 및 업로드 수행
"""

import argparse
import os
import queue
import re
import signal
import sys
import time
from datetime import datetime
from typing import Dict

import cv2
import requests

import config
from metrics import EdgeMetrics, format_ms
from preprocessor import PCBPreprocessor
from rtsp_receiver import RTSPReceiver
from scavenger_worker import ScavengerWorker
from inference_worker import InferenceWorker
from upload_worker import UploadWorker


def _dict_delta(current: Dict[str, int], previous: Dict[str, int]) -> Dict[str, int]:
    keys = set(current.keys()) | set(previous.keys())
    return {key: current.get(key, 0) - previous.get(key, 0) for key in keys}


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _expand_channel_sources(base_source: str, num_cameras: int) -> list[str]:
    """
    base_source를 channel suffix 규칙(_1, _2, ...)으로 확장한다.
    예: rtsp://IP:8554/pcb_stream -> rtsp://IP:8554/pcb_stream_1 ...
    """
    head, has_query, query = base_source.partition("?")
    # 이미 _숫자 suffix가 있으면 base로 정규화 후 재생성
    head = re.sub(r"_\d+$", "", head)
    suffix = f"?{query}" if has_query else ""
    return [f"{head}_{i+1}{suffix}" for i in range(num_cameras)]


def _compute_queue_sizes(num_cameras: int):
    """P1-2: 카메라 수 기반 동적 큐 스케일링."""
    if not config.QUEUE_DYNAMIC_SCALING_ENABLED:
        return config.FRAME_QUEUE_SIZE, config.CROP_QUEUE_SIZE, config.UPLOAD_QUEUE_SIZE

    frame_target = int(round(config.QUEUE_TARGET_INPUT_FPS * config.FRAME_QUEUE_BUFFER_SEC))
    frame_size = _clamp(frame_target, config.FRAME_QUEUE_MIN_SIZE, config.FRAME_QUEUE_MAX_SIZE)
    crop_size = _clamp(
        config.CROP_QUEUE_SIZE + config.CROP_QUEUE_PER_CAMERA * max(0, num_cameras - 1),
        config.CROP_QUEUE_SIZE,
        config.CROP_QUEUE_MAX_SIZE,
    )
    upload_size = _clamp(
        config.UPLOAD_QUEUE_SIZE + config.UPLOAD_QUEUE_PER_CAMERA * max(0, num_cameras - 1),
        config.UPLOAD_QUEUE_SIZE,
        config.UPLOAD_QUEUE_MAX_SIZE,
    )
    return frame_size, crop_size, upload_size


def _next_queue_alert_level(ratio: float, current_level: str) -> str:
    if ratio >= config.QUEUE_CRIT_RATIO:
        return "crit"
    if ratio >= config.QUEUE_WARN_RATIO:
        return "warn"
    if ratio <= config.QUEUE_RECOVERY_RATIO:
        return "ok"
    return current_level


def _build_queue_alerts(snapshot: dict, previous_snapshot: dict, queue_capacities: dict, queue_alert_state: dict):
    """
    P1-2: Queue watermark 경보 고도화.
    - ratio 기반 warn/crit/recovery
    - hold time 기반 노이즈 억제
    - interval drop 증가 감지
    """
    now = snapshot["ts"]
    alerts = []
    current_depths = snapshot.get("queue_current_depth", {})
    current_drops = snapshot.get("queue_drops", {})
    previous_drops = previous_snapshot.get("queue_drops", {})

    for queue_name, maxsize in queue_capacities.items():
        if maxsize <= 0:
            continue

        depth = current_depths.get(queue_name, 0)
        ratio = depth / maxsize
        state = queue_alert_state.setdefault(
            queue_name,
            {
                "level": "ok",
                "since": now,
                "last_emitted_level": "ok",
            },
        )
        prev_level = state["level"]
        next_level = _next_queue_alert_level(ratio, prev_level)
        if next_level != prev_level:
            state["level"] = next_level
            state["since"] = now
            if next_level == "ok":
                if prev_level in ("warn", "crit"):
                    alerts.append(
                        f"  queue_alert[RECOVERY] {queue_name}: "
                        f"depth={depth}/{maxsize} ({ratio*100:.1f}%)"
                    )
                state["last_emitted_level"] = "ok"

        hold_sec = now - state["since"]
        if state["level"] == "warn":
            if hold_sec >= config.QUEUE_WARN_HOLD_SEC and state["last_emitted_level"] != "warn":
                alerts.append(
                    f"  queue_alert[WARN] {queue_name}: "
                    f"depth={depth}/{maxsize} ({ratio*100:.1f}%), hold={hold_sec:.1f}s"
                )
                state["last_emitted_level"] = "warn"
        elif state["level"] == "crit":
            if hold_sec >= config.QUEUE_CRIT_HOLD_SEC and state["last_emitted_level"] != "crit":
                alerts.append(
                    f"  queue_alert[CRIT] {queue_name}: "
                    f"depth={depth}/{maxsize} ({ratio*100:.1f}%), hold={hold_sec:.1f}s"
                )
                state["last_emitted_level"] = "crit"

        drop_delta = current_drops.get(queue_name, 0) - previous_drops.get(queue_name, 0)
        if drop_delta > 0:
            alerts.append(f"  queue_alert[DROP] {queue_name}: +{drop_delta} drops (interval)")

    return alerts


def _diagnose_bottleneck(snapshot: dict, upload_queue_maxsize: int) -> str:
    upload_wait_p95 = snapshot["latency"]["upload_queue_wait_ms"]["p95"] or 0.0
    upload_p95 = snapshot["latency"]["upload_ms"]["p95"] or 0.0
    inference_p95 = snapshot["latency"]["inference_ms"]["p95"] or 0.0
    frame_wait_p95 = snapshot["latency"]["frame_queue_wait_ms"]["p95"] or 0.0
    preprocess_p95 = snapshot["latency"]["preprocess_ms"]["p95"] or 0.0

    upload_hwm = snapshot["queue_high_watermark"].get("upload_queue", 0)
    frame_hwm = snapshot["queue_high_watermark"].get("frame_queue", 0)

    upload_suspect = (
        upload_hwm >= max(1, int(upload_queue_maxsize * 0.9))
        or upload_wait_p95 >= 1000.0
        or upload_p95 >= 1000.0
    )
    if upload_suspect and upload_wait_p95 > max(inference_p95 * 2.0, 200.0):
        return "Upload path bottleneck suspected (network/backend/upload queue)"

    if frame_hwm >= 1 and frame_wait_p95 > max(preprocess_p95 * 2.0, 30.0):
        return "Frame ingest/preprocess bottleneck suspected"

    return "No dominant bottleneck signal (balanced or insufficient evidence)"


def _pull_next_frame_round_robin(
    camera_ids,
    frame_queues,
    rr_cursor: int,
):
    """
    camera별 frame queue를 라운드로빈으로 순회하며 다음 프레임을 가져온다.
    Returns:
      item, next_cursor, total_depth
    """
    if not camera_ids:
        return None, rr_cursor, 0

    n = len(camera_ids)
    total_depth = sum(frame_queues[cid].qsize() for cid in camera_ids)
    for offset in range(n):
        idx = (rr_cursor + offset) % n
        camera_id = camera_ids[idx]
        q = frame_queues[camera_id]
        try:
            item = q.get_nowait()
            next_cursor = (idx + 1) % n
            return item, next_cursor, total_depth
        except queue.Empty:
            continue

    return None, rr_cursor, total_depth


def _print_periodic_metrics(
    metrics: EdgeMetrics,
    previous_snapshot: dict,
    queue_capacities: dict,
    queue_alert_state: dict,
) -> dict:
    snapshot = metrics.snapshot(config.FAILED_DIR)
    interval = max(snapshot["ts"] - previous_snapshot["ts"], 1e-6)

    delta_input = _dict_delta(snapshot["input_frames"], previous_snapshot["input_frames"])
    delta_processed = _dict_delta(snapshot["processed_frames"], previous_snapshot["processed_frames"])
    delta_inference = _dict_delta(snapshot["inference_count"], previous_snapshot["inference_count"])
    delta_upload_success = snapshot["upload_success"] - previous_snapshot["upload_success"]
    delta_upload_fail = snapshot["upload_fail"] - previous_snapshot["upload_fail"]
    delta_scavenger_success = snapshot["scavenger_success"] - previous_snapshot["scavenger_success"]

    total_input_fps = (snapshot["totals"]["input_frames"] - previous_snapshot["totals"]["input_frames"]) / interval
    total_processed_fps = (
        snapshot["totals"]["processed_frames"] - previous_snapshot["totals"]["processed_frames"]
    ) / interval
    total_infer_fps = (snapshot["totals"]["inference"] - previous_snapshot["totals"]["inference"]) / interval

    print("\n[Metrics] === 5s Summary ===")
    print(
        f"  throughput(total): input={total_input_fps:.2f}fps, "
        f"processed={total_processed_fps:.2f}fps, inference={total_infer_fps:.2f}crops/s"
    )
    for camera_id in sorted(set(delta_input.keys()) | set(delta_processed.keys()) | set(delta_inference.keys())):
        input_fps = delta_input.get(camera_id, 0) / interval
        proc_fps = delta_processed.get(camera_id, 0) / interval
        inf_fps = delta_inference.get(camera_id, 0) / interval
        total_input = snapshot["input_frames"].get(camera_id, 0)
        total_drop = snapshot["input_drops"].get(camera_id, 0)
        drop_rate = (total_drop / total_input * 100.0) if total_input > 0 else 0.0
        print(
            f"  [{camera_id}] input={input_fps:.2f}fps, processed={proc_fps:.2f}fps, "
            f"infer={inf_fps:.2f}crops/s, drop={drop_rate:.2f}% ({total_drop}/{total_input})"
        )

    interval_live_attempt = delta_upload_success + delta_upload_fail
    interval_live_success_rate = _safe_rate(delta_upload_success * 100.0, interval_live_attempt)
    cumulative_inference = snapshot["totals"]["inference"]
    cumulative_effective_delivery = snapshot["upload_success"] + snapshot["scavenger_success"]
    cumulative_effective_delivery_rate = _safe_rate(cumulative_effective_delivery * 100.0, cumulative_inference)

    print(
        f"  upload(live): success={snapshot['upload_success']}, fail={snapshot['upload_fail']} "
        f"(interval success={interval_live_success_rate:.2f}%)"
    )
    print(
        f"  delivery(effective): {(cumulative_effective_delivery_rate):.2f}% "
        f"(live+scavenger success={cumulative_effective_delivery} / inference={cumulative_inference}), "
        f"backlog={snapshot['failed_backlog']}"
    )
    print(
        f"  scavenger: success={snapshot['scavenger_success']} (+{delta_scavenger_success}), "
        f"fail={snapshot['scavenger_fail']}, expired={snapshot['scavenger_expired']}"
    )
    print(
        "  queue_hwm: "
        f"frame={snapshot['queue_high_watermark'].get('frame_queue', 0)}, "
        f"crop={snapshot['queue_high_watermark'].get('crop_queue', 0)}, "
        f"upload={snapshot['queue_high_watermark'].get('upload_queue', 0)} "
        f"| queue_drop(frame/crop/upload)="
        f"{snapshot['queue_drops'].get('frame_queue', 0)}/"
        f"{snapshot['queue_drops'].get('crop_queue', 0)}/"
        f"{snapshot['queue_drops'].get('upload_queue', 0)}"
    )
    print(
        "  queue_depth(now): "
        f"frame={snapshot['queue_current_depth'].get('frame_queue', 0)}, "
        f"crop={snapshot['queue_current_depth'].get('crop_queue', 0)}, "
        f"upload={snapshot['queue_current_depth'].get('upload_queue', 0)}"
    )

    frame_hwm_by_cam = {
        key.replace("frame_queue_", ""): value
        for key, value in snapshot["queue_high_watermark"].items()
        if key.startswith("frame_queue_")
    }
    if frame_hwm_by_cam:
        hwm_text = ", ".join(f"{cam}:{depth}" for cam, depth in sorted(frame_hwm_by_cam.items()))
        print(f"  frame_queue_hwm_by_cam: {hwm_text}")

    frame_drop_by_cam = {
        key.replace("frame_queue_", ""): value
        for key, value in snapshot["queue_drops"].items()
        if key.startswith("frame_queue_")
    }
    if frame_drop_by_cam:
        drop_text = ", ".join(f"{cam}:{cnt}" for cam, cnt in sorted(frame_drop_by_cam.items()))
        print(f"  frame_queue_drop_by_cam: {drop_text}")

    print(
        "  latency(ms) p50/p95: "
        f"frame_wait={format_ms(snapshot['latency']['frame_queue_wait_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['frame_queue_wait_ms']['p95'])}, "
        f"preprocess={format_ms(snapshot['latency']['preprocess_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['preprocess_ms']['p95'])}, "
        f"crop_wait={format_ms(snapshot['latency']['crop_queue_wait_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['crop_queue_wait_ms']['p95'])}, "
        f"inference={format_ms(snapshot['latency']['inference_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['inference_ms']['p95'])}, "
        f"upload_wait={format_ms(snapshot['latency']['upload_queue_wait_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['upload_queue_wait_ms']['p95'])}, "
        f"upload={format_ms(snapshot['latency']['upload_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['upload_ms']['p95'])}, "
        f"e2e={format_ms(snapshot['latency']['e2e_ms']['p50'])}/"
        f"{format_ms(snapshot['latency']['e2e_ms']['p95'])}"
    )
    print(
        "  upload_payload(kb) p50/p95: "
        f"{format_ms(snapshot['latency']['upload_payload_kb']['p50'])}/"
        f"{format_ms(snapshot['latency']['upload_payload_kb']['p95'])}"
    )
    alerts = _build_queue_alerts(snapshot, previous_snapshot, queue_capacities, queue_alert_state)
    if alerts:
        for alert in alerts:
            print(alert)
    print(
        f"  diagnosis: "
        f"{_diagnose_bottleneck(snapshot, queue_capacities.get('upload_queue', config.UPLOAD_QUEUE_SIZE))}"
    )
    return snapshot


def start_session(session_url: str) -> int:
    """
    백엔드에 세션 시작을 요청하고 세션 ID를 반환.
    실패 시 None 반환.
    """
    try:
        response = requests.post(session_url, timeout=5.0)
        if response.status_code == 201:
            data = response.json()
            session_id = data.get("id")
            print(f"[Main] 세션 시작: ID={session_id}")
            return session_id
        else:
            print(f"[Main] 세션 시작 실패: HTTP {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[Main] 세션 시작 요청 실패: {e}")
        return None


def end_session(session_url: str, session_id: int):
    """
    백엔드에 세션 종료를 요청.
    """
    if session_id is None:
        return

    try:
        response = requests.patch(f"{session_url}{session_id}", timeout=5.0)
        if response.status_code == 200:
            print(f"[Main] 세션 종료: ID={session_id}")
        else:
            print(f"[Main] 세션 종료 실패: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[Main] 세션 종료 요청 실패: {e}")


def save_crop_for_debug(crop, output_dir: str, index: int):
    """디버그용 크롭 이미지 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"crop_{index:04d}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, crop)
    print(f"[Debug] 크롭 저장: {filepath}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="PCB 전처리 및 추론 파이프라인")
    parser.add_argument(
        "--input", "-i",
        default=config.RTSP_URL,
        help=f"RTSP URL 또는 비디오 파일 경로 (기본값: {config.RTSP_URL})"
    )
    parser.add_argument(
        "--api-url", "-a",
        default=config.API_URL,
        help=f"백엔드 API 주소 (기본값: {config.API_URL})"
    )
    parser.add_argument(
        "--model", "-m",
        default=config.MODEL_PATH,
        help=f"YOLO 모델 경로 (기본값: {config.MODEL_PATH})"
    )
    parser.add_argument(
        "--loop", "-l",
        action="store_true",
        help="비디오 파일 반복 재생 (테스트용)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="디버그 모드 (크롭 이미지 저장)"
    )
    parser.add_argument(
        "--session-url",
        default="http://3.35.182.98:8080/sessions/",
        help="세션 API 주소"
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="세션 관리 비활성화"
    )
    parser.add_argument(
        "--no-scavenger",
        action="store_true",
        help="실패 파일 자동 재전송 워커 비활성화"
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="최대 크롭 개수 (0=무제한)"
    )
    parser.add_argument(
        "--num-cameras", "-n",
        type=int,
        default=1,
        help="사용할 카메라 대수 (기본값: 1)"
    )
    parser.add_argument(
        "--capture-backend",
        choices=["auto", "gstreamer", "ffmpeg"],
        default=config.RTSP_CAPTURE_BACKEND,
        help=f"RTSP 캡처 백엔드 (기본값: {config.RTSP_CAPTURE_BACKEND})"
    )
    parser.add_argument(
        "--no-hw-decode",
        action="store_true",
        help="RTSP HW decode(nvv4l2decoder) 비활성화"
    )
    parser.add_argument(
        "--rtsp-latency-ms",
        type=int,
        default=config.RTSP_LATENCY_MS,
        help=f"GStreamer RTSP latency(ms) (기본값: {config.RTSP_LATENCY_MS})"
    )
    parser.add_argument(
        "--no-rtsp-reconnect",
        action="store_true",
        help="RTSP 자동 재연결 비활성화"
    )
    
    parser.add_argument(
        "--upload-transport",
        choices=["multipart", "json"],
        default=config.UPLOAD_TRANSPORT,
        help=f"업로드 전송 포맷 (기본값: {config.UPLOAD_TRANSPORT})"
    )
    parser.add_argument(
        "--upload-jpeg-quality",
        type=int,
        default=config.UPLOAD_IMAGE_JPEG_QUALITY,
        help=f"업로드 JPEG 품질 1~100 (기본값: {config.UPLOAD_IMAGE_JPEG_QUALITY})"
    )
    parser.add_argument(
        "--upload-max-width",
        type=int,
        default=config.UPLOAD_IMAGE_MAX_WIDTH,
        help=f"업로드 이미지 최대 너비(px, 0=원본 유지) (기본값: {config.UPLOAD_IMAGE_MAX_WIDTH})"
    )
    parser.add_argument(
        "--upload-max-height",
        type=int,
        default=config.UPLOAD_IMAGE_MAX_HEIGHT,
        help=f"업로드 이미지 최대 높이(px, 0=원본 유지) (기본값: {config.UPLOAD_IMAGE_MAX_HEIGHT})"
    )
    parser.add_argument(
        "--upload-grayscale",
        action="store_true",
        help="업로드 이미지를 grayscale로 인코딩"
    )

    args = parser.parse_args()

    # 입력 소스 처리
    raw_sources = [s.strip() for s in args.input.split(",") if s.strip()]
    num_cameras_flag_used = ("-n" in sys.argv) or ("--num-cameras" in sys.argv)

    if len(raw_sources) > 1:
        # 쉼표로 여러 소스를 직접 지정한 경우 그대로 사용
        input_sources = raw_sources
    else:
        base_url = raw_sources[0] if raw_sources else args.input.strip()
        is_video_file = any(base_url.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv"])

        if is_video_file:
            # 파일인 경우 동일 파일을 n번 수신 (테스트용)
            input_sources = [base_url] * args.num_cameras if args.num_cameras > 1 else [base_url]
        elif args.num_cameras > 1 or (args.num_cameras == 1 and num_cameras_flag_used):
            # -n 옵션 사용 시 1대도 _1 suffix로 정규화
            input_sources = _expand_channel_sources(base_url, args.num_cameras)
        else:
            # 기존 단일 URL 동작 유지
            input_sources = [base_url]

    # 동적 설정 업데이트 (CLI 인자 우선)
    config.API_URL = args.api_url
    config.MODEL_PATH = args.model
    config.RTSP_CAPTURE_BACKEND = args.capture_backend
    config.RTSP_HW_DECODE = not args.no_hw_decode
    config.RTSP_LATENCY_MS = max(0, args.rtsp_latency_ms)
    config.RTSP_RECONNECT_ENABLED = not args.no_rtsp_reconnect
    config.UPLOAD_TRANSPORT = args.upload_transport
    config.UPLOAD_IMAGE_JPEG_QUALITY = max(1, min(100, int(args.upload_jpeg_quality)))
    config.UPLOAD_IMAGE_MAX_WIDTH = max(0, int(args.upload_max_width))
    config.UPLOAD_IMAGE_MAX_HEIGHT = max(0, int(args.upload_max_height))
    config.UPLOAD_IMAGE_GRAYSCALE = bool(args.upload_grayscale)

    # 배경 이미지 확인
    if not os.path.exists(config.BACKGROUND_PATH):
        print(f"[Main] 에러: 배경 이미지({config.BACKGROUND_PATH})가 없습니다.")
        sys.exit(1)

    # 세션 시작
    session_id = None
    if not args.no_session:
        session_id = start_session(args.session_url)

    # Queue 생성
    # P1-1: camera별 frame queue 분리 (공정 스케줄링 기반)
    # P1-2: camera 수 기반 동적 큐 스케일링
    frame_queue_size, crop_queue_size, upload_queue_size = _compute_queue_sizes(len(input_sources))
    frame_queues = {}
    crop_queue = queue.Queue(maxsize=crop_queue_size)
    upload_queue = queue.Queue(maxsize=upload_queue_size)
    metrics = EdgeMetrics(latency_buffer_size=config.METRICS_LATENCY_BUFFER_SIZE)
    queue_capacities = {
        "frame_queue": frame_queue_size * len(input_sources),
        "crop_queue": crop_queue_size,
        "upload_queue": upload_queue_size,
    }

    # 1. 업로드 워커 시작
    upload_worker = UploadWorker(upload_queue, metrics=metrics, api_url=config.API_URL)
    upload_worker.start()

    # 1-1. 재전송 워커 시작
    scavenger_worker = None
    if config.SCAVENGER_ENABLED and not args.no_scavenger:
        scavenger_worker = ScavengerWorker(
            failed_dir=config.FAILED_DIR,
            api_url=config.API_URL,
            metrics=metrics,
            poll_interval_sec=config.SCAVENGER_POLL_INTERVAL_SEC,
            base_backoff_sec=config.SCAVENGER_BASE_BACKOFF_SEC,
            max_backoff_sec=config.SCAVENGER_MAX_BACKOFF_SEC,
            jitter_ratio=config.SCAVENGER_JITTER_RATIO,
            max_retries=config.SCAVENGER_MAX_RETRIES,
            ttl_sec=config.SCAVENGER_TTL_SEC,
        )
        scavenger_worker.start()

    # 2. 추론 워커 초기화
    print(f"[Main] 추론 워커 초기화 중 (모델: {config.MODEL_PATH})...")
    try:
        inference_worker = InferenceWorker(
            crop_queue,
            upload_queue,
            config.MODEL_PATH,
            session_id=session_id,
            metrics=metrics,
        )
    except Exception as e:
        print(f"[Main] 추론 워커 초기화 실패: {e}")
        upload_worker.stop()
        upload_worker.join(timeout=2.0)
        if scavenger_worker:
            scavenger_worker.stop()
            scavenger_worker.join(timeout=2.0)
        if session_id:
            end_session(args.session_url, session_id)
        sys.exit(1)

    # 3. RTSP 수신 스레드 리스트 시작
    receivers = []
    for i, source in enumerate(input_sources):
        camera_id = f"cam_{i+1}"
        frame_queue = queue.Queue(maxsize=frame_queue_size)
        frame_queues[camera_id] = frame_queue
        queue_capacities[f"frame_queue_{camera_id}"] = frame_queue_size
        receiver = RTSPReceiver(
            source,
            frame_queue,
            camera_id=camera_id,
            loop=args.loop,
            metrics=metrics,
            queue_name=f"frame_queue_{camera_id}",
            capture_backend=config.RTSP_CAPTURE_BACKEND,
            use_hw_decode=config.RTSP_HW_DECODE,
            rtsp_latency_ms=config.RTSP_LATENCY_MS,
            reconnect_enabled=config.RTSP_RECONNECT_ENABLED,
            reconnect_base_delay_sec=config.RTSP_RECONNECT_BASE_SEC,
            reconnect_max_delay_sec=config.RTSP_RECONNECT_MAX_SEC,
            max_failures=config.RTSP_READ_FAIL_THRESHOLD,
        )
        receiver.start()
        receivers.append(receiver)

    # 연결 대기 (최대 10초)
    print(f"[Main] {len(receivers)}개 RTSP 연결 대기 중...")
    start_wait = time.time()
    while time.time() - start_wait < 10:
        if all(r.is_running() for r in receivers):
            break
        time.sleep(0.1)
    
    # 4. 추론 워커 가동
    inference_worker.start()

    # 카메라별 전처리기 초기화 (상태 오염 방지)
    preprocessors = {
        receiver.camera_id: PCBPreprocessor(config.BACKGROUND_PATH)
        for receiver in receivers
    }

    # 종료 시그널 핸들러
    shutdown_flag = False
    def signal_handler(signum, frame):
        nonlocal shutdown_flag
        print("\n[Main] 종료 신호 수신...")
        shutdown_flag = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[Main] 파이프라인 가동 시작")
    print(f"  - 입력 소스: {len(input_sources)}개")
    for r in receivers:
        print(f"    * [{r.camera_id}] {r.source}")
    print(f"  - API: {config.API_URL}")
    print(f"  - 모델: {config.MODEL_PATH}")
    print(
        "  - Queue size: "
        f"frame_per_cam={frame_queue_size}, "
        f"crop={crop_queue_size}, upload={upload_queue_size} "
        f"(dynamic={'on' if config.QUEUE_DYNAMIC_SCALING_ENABLED else 'off'})"
    )
    print(
        "  - RTSP capture: "
        f"backend={config.RTSP_CAPTURE_BACKEND}, "
        f"hw_decode={'on' if config.RTSP_HW_DECODE else 'off'}, "
        f"latency={config.RTSP_LATENCY_MS}ms, "
        f"reconnect={'on' if config.RTSP_RECONNECT_ENABLED else 'off'}"
    )
    print(
        "  - Upload image: "
        f"transport={config.UPLOAD_TRANSPORT}, "
        f"jpeg_quality={config.UPLOAD_IMAGE_JPEG_QUALITY}, "
        f"max={config.UPLOAD_IMAGE_MAX_WIDTH}x{config.UPLOAD_IMAGE_MAX_HEIGHT}, "
        f"grayscale={'on' if config.UPLOAD_IMAGE_GRAYSCALE else 'off'}"
    )
    print(f"  - 세션 ID: {session_id if session_id else '없음'}")
    print(f"  - Scavenger: {'활성화' if scavenger_worker else '비활성화'}")

    crop_count = 0
    frame_count = 0
    start_time = time.time()
    last_metrics_ts = start_time
    last_metrics_snapshot = metrics.snapshot(config.FAILED_DIR)
    queue_alert_state = {}
    camera_ids = [r.camera_id for r in receivers]
    rr_cursor = 0

    try:
        while not shutdown_flag:
            # camera별 frame queue를 라운드로빈으로 순회하여 공정 소비
            frame_item, rr_cursor, total_frame_depth = _pull_next_frame_round_robin(
                camera_ids,
                frame_queues,
                rr_cursor,
            )
            metrics.update_queue_depth("frame_queue", total_frame_depth)
            if frame_item is None:
                all_receivers_stopped = all(not r.is_alive() for r in receivers)
                all_frame_queues_empty = all(frame_queues[cid].empty() for cid in camera_ids)
                if all_receivers_stopped and all_frame_queues_empty:
                    break
                time.sleep(0.002)
                continue

            camera_id = "unknown"
            frame = None
            frame_ts = None

            if isinstance(frame_item, tuple):
                if len(frame_item) == 3:
                    camera_id, frame, frame_ts = frame_item
                elif len(frame_item) >= 2:
                    camera_id, frame = frame_item[0], frame_item[1]
            if frame is None:
                continue

            camera_queue = frame_queues.get(camera_id)
            if camera_queue is not None:
                metrics.update_queue_depth(f"frame_queue_{camera_id}", camera_queue.qsize())

            frame_count += 1
            metrics.record_processed(camera_id)
            if frame_ts:
                metrics.record_latency("frame_queue_wait_ms", (time.time() - frame_ts) * 1000.0)

            preprocessor = preprocessors.get(camera_id)
            if preprocessor is None:
                # 예상치 못한 camera_id가 들어와도 상태를 분리해 처리
                preprocessor = PCBPreprocessor(config.BACKGROUND_PATH)
                preprocessors[camera_id] = preprocessor
                print(f"[Main] 신규 카메라 전처리기 생성: {camera_id}")

            preprocess_start = time.time()
            cropped = preprocessor.process_frame(frame)
            preprocess_ms = (time.time() - preprocess_start) * 1000.0
            metrics.record_latency("preprocess_ms", preprocess_ms)

            if cropped is not None:
                crop_count += 1
                metrics.record_crop(camera_id)
                print(f"[Main][{camera_id}] [#{crop_count}] PCB 포착! ({cropped.shape[1]}x{cropped.shape[0]})")

                crop_item = {
                    "camera_id": camera_id,
                    "crop": cropped,
                    "frame_ts": frame_ts,
                    "preprocess_done_ts": time.time(),
                }
                try:
                    crop_queue.put_nowait(crop_item)
                    metrics.update_queue_depth("crop_queue", crop_queue.qsize())
                except queue.Full:
                    try:
                        crop_queue.get_nowait()
                        metrics.record_queue_drop("crop_queue")
                        crop_queue.put_nowait(crop_item)
                        metrics.update_queue_depth("crop_queue", crop_queue.qsize())
                    except queue.Empty:
                        pass

                if args.debug:
                    save_crop_for_debug(cropped, config.DEBUG_DIR, crop_count)

                if args.max_crops > 0 and crop_count >= args.max_crops:
                    break

            now = time.time()
            if now - last_metrics_ts >= config.METRICS_LOG_INTERVAL_SEC:
                last_metrics_snapshot = _print_periodic_metrics(
                    metrics,
                    last_metrics_snapshot,
                    queue_capacities,
                    queue_alert_state,
                )
                last_metrics_ts = now

    except Exception as e:
        print(f"[Main] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Main] 리소스 정리 중...")
        for r in receivers:
            r.stop()
        inference_worker.stop()
        upload_worker.stop()
        if scavenger_worker:
            scavenger_worker.stop()

        for r in receivers:
            r.join(timeout=2.0)
        inference_worker.join(timeout=2.0)
        upload_worker.join(timeout=2.0)
        if scavenger_worker:
            scavenger_worker.join(timeout=2.0)

        if session_id:
            end_session(args.session_url, session_id)

        elapsed = time.time() - start_time
        aggregate_processed_fps = frame_count / elapsed if elapsed > 0 else 0.0
        per_camera_avg_fps = aggregate_processed_fps / len(receivers) if receivers else 0.0

        print("\n[Main] === 최종 통계 ===")
        print(f"  처리 프레임: {frame_count}")
        print(f"  포착된 PCB: {crop_count}")
        print(f"  실행 시간: {elapsed:.1f}초")
        print(f"  총 처리량(Aggregate Processed FPS): {aggregate_processed_fps:.1f}")
        print(f"  카메라당 평균 처리 FPS(참고): {per_camera_avg_fps:.1f}")
        for r in receivers:
            print(f"  [{r.camera_id}] 상태: {r.get_stats()}")
        print(f"  세션 ID: {session_id if session_id else '없음'}")

        final_snapshot = metrics.snapshot(config.FAILED_DIR)
        print("\n[Metrics] === Final Summary ===")
        for camera_id in sorted(final_snapshot["input_frames"].keys()):
            total_input = final_snapshot["input_frames"].get(camera_id, 0)
            total_drop = final_snapshot["input_drops"].get(camera_id, 0)
            processed = final_snapshot["processed_frames"].get(camera_id, 0)
            inferred = final_snapshot["inference_count"].get(camera_id, 0)
            drop_rate = (total_drop / total_input * 100.0) if total_input > 0 else 0.0
            input_fps_avg = _safe_rate(total_input, elapsed)
            processed_fps_avg = _safe_rate(processed, elapsed)
            print(
                f"  [{camera_id}] input={total_input} ({input_fps_avg:.2f}fps), "
                f"processed={processed} ({processed_fps_avg:.2f}fps), "
                f"inference={inferred}, drop={total_drop} ({drop_rate:.2f}%)"
            )

        cumulative_inference = final_snapshot["totals"]["inference"]
        live_success = final_snapshot["upload_success"]
        scavenger_success = final_snapshot["scavenger_success"]
        effective_success = live_success + scavenger_success
        live_delivery_rate = _safe_rate(live_success * 100.0, cumulative_inference)
        effective_delivery_rate = _safe_rate(effective_success * 100.0, cumulative_inference)

        print(
            f"  upload(live): success={live_success}, fail={final_snapshot['upload_fail']}, "
            f"live_delivery_rate={live_delivery_rate:.2f}%"
        )
        print(
            f"  scavenger_success={final_snapshot['scavenger_success']}, "
            f"scavenger_fail={final_snapshot['scavenger_fail']}, "
            f"scavenger_expired={final_snapshot['scavenger_expired']}"
        )
        print(
            f"  delivery(effective): {effective_success}/{cumulative_inference} "
            f"({effective_delivery_rate:.2f}%), backlog={final_snapshot['failed_backlog']}"
        )
        print(
            "  queue_hwm: "
            f"frame={final_snapshot['queue_high_watermark'].get('frame_queue', 0)}, "
            f"crop={final_snapshot['queue_high_watermark'].get('crop_queue', 0)}, "
            f"upload={final_snapshot['queue_high_watermark'].get('upload_queue', 0)} "
            f"| queue_drop(frame/crop/upload)="
            f"{final_snapshot['queue_drops'].get('frame_queue', 0)}/"
            f"{final_snapshot['queue_drops'].get('crop_queue', 0)}/"
            f"{final_snapshot['queue_drops'].get('upload_queue', 0)}"
        )
        print(
            "  queue_depth(now): "
            f"frame={final_snapshot['queue_current_depth'].get('frame_queue', 0)}, "
            f"crop={final_snapshot['queue_current_depth'].get('crop_queue', 0)}, "
            f"upload={final_snapshot['queue_current_depth'].get('upload_queue', 0)}"
        )
        frame_hwm_by_cam = {
            key.replace("frame_queue_", ""): value
            for key, value in final_snapshot["queue_high_watermark"].items()
            if key.startswith("frame_queue_")
        }
        if frame_hwm_by_cam:
            hwm_text = ", ".join(f"{cam}:{depth}" for cam, depth in sorted(frame_hwm_by_cam.items()))
            print(f"  frame_queue_hwm_by_cam: {hwm_text}")

        frame_drop_by_cam = {
            key.replace("frame_queue_", ""): value
            for key, value in final_snapshot["queue_drops"].items()
            if key.startswith("frame_queue_")
        }
        if frame_drop_by_cam:
            drop_text = ", ".join(f"{cam}:{cnt}" for cam, cnt in sorted(frame_drop_by_cam.items()))
            print(f"  frame_queue_drop_by_cam: {drop_text}")

        print(
            "  latency(ms) p50/p95: "
            f"frame_wait={format_ms(final_snapshot['latency']['frame_queue_wait_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['frame_queue_wait_ms']['p95'])}, "
            f"preprocess={format_ms(final_snapshot['latency']['preprocess_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['preprocess_ms']['p95'])}, "
            f"crop_wait={format_ms(final_snapshot['latency']['crop_queue_wait_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['crop_queue_wait_ms']['p95'])}, "
            f"inference={format_ms(final_snapshot['latency']['inference_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['inference_ms']['p95'])}, "
            f"upload_wait={format_ms(final_snapshot['latency']['upload_queue_wait_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['upload_queue_wait_ms']['p95'])}, "
            f"upload={format_ms(final_snapshot['latency']['upload_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['upload_ms']['p95'])}, "
            f"e2e={format_ms(final_snapshot['latency']['e2e_ms']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['e2e_ms']['p95'])}"
        )
        print(
            "  upload_payload(kb) p50/p95: "
            f"{format_ms(final_snapshot['latency']['upload_payload_kb']['p50'])}/"
            f"{format_ms(final_snapshot['latency']['upload_payload_kb']['p95'])}"
        )
        print(
            f"  diagnosis: "
            f"{_diagnose_bottleneck(final_snapshot, queue_capacities.get('upload_queue', upload_queue_size))}"
        )


if __name__ == "__main__":
    main()
