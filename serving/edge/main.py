"""
PCB 전처리 파이프라인 메인 모듈

RTSP 영상을 수신하고, PCB를 감지/크롭하여 추론 및 업로드 수행
"""

import argparse
import os
import queue
import signal
import sys
import time
from datetime import datetime

import cv2

import config
from preprocessor import PCBPreprocessor
from rtsp_receiver import RTSPReceiver
from inference_worker import InferenceWorker
from upload_worker import UploadWorker


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
    parser.add_argument("--input", "-i", default=config.RTSP_URL, help="RTSP URL 또는 비디오 파일 경로")
    parser.add_argument("--api-url", "-a", default=config.API_URL, help="백엔드 API 주소")
    parser.add_argument("--model", "-m", default=config.MODEL_PATH, help="YOLO 모델 경로")
    parser.add_argument("--loop", "-l", action="store_true", help="비디오 파일 반복 재생")
    parser.add_argument("--debug", "-d", action="store_true", help="디버그 모드 (크롭 이미지 저장)")
    parser.add_argument("--max-crops", type=int, default=0, help="최대 크롭 개수 (0=무제한)")
    args = parser.parse_args()

    # 동적 설정 업데이트 (CLI 인자 우선)
    config.RTSP_URL = args.input
    config.API_URL = args.api_url
    config.MODEL_PATH = args.model

    # 배경 이미지 확인
    if not os.path.exists(config.BACKGROUND_PATH):
        print(f"[Main] 에러: 배경 이미지({config.BACKGROUND_PATH})가 없습니다.")
        sys.exit(1)

    # Queue 생성
    frame_queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
    crop_queue = queue.Queue(maxsize=config.CROP_QUEUE_SIZE)
    upload_queue = queue.Queue(maxsize=config.UPLOAD_QUEUE_SIZE)

    # 1. 업로드 워커 시작
    upload_worker = UploadWorker(upload_queue)
    upload_worker.start()

    # 2. 추론 워커 시작 (모델 로드 및 엔진 변환 수행)
    print(f"[Main] 추론 워커 초기화 중 (모델: {config.MODEL_PATH})...")
    try:
        inference_worker = InferenceWorker(crop_queue, upload_queue)
    except Exception as e:
        print(f"[Main] 추론 워커 초기화 실패: {e}")
        upload_worker.stop()
        sys.exit(1)

    # 3. RTSP 수신 스레드 시작
    receiver = RTSPReceiver(config.RTSP_URL, frame_queue, loop=args.loop)
    receiver.start()

    # 연결 대기 (최대 10초)
    print("[Main] RTSP 연결 대기 중...")
    for _ in range(100):
        if receiver.is_running():
            break
        if not receiver.is_alive():
            print("[Main] 수신 스레드가 시작되지 않았습니다.")
            sys.exit(1)
        time.sleep(0.1)
    
    # 4. 모든 준비 완료 후 추론 워커 가동
    inference_worker.start()

    # 전처리기 초기화
    preprocessor = PCBPreprocessor(config.BACKGROUND_PATH)

    # 종료 시그널 핸들러
    shutdown_flag = False
    def signal_handler(signum, frame):
        nonlocal shutdown_flag
        print("\n[Main] 종료 신호 수신...")
        shutdown_flag = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[Main] 파이프라인 가동 시작")
    crop_count = 0
    frame_count = 0
    start_time = time.time()

    try:
        while not shutdown_flag:
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not receiver.is_running():
                    break
                continue

            frame_count += 1
            cropped = preprocessor.process_frame(frame)

            if cropped is not None:
                crop_count += 1
                print(f"[Main] [#{crop_count}] PCB 포착! ({cropped.shape[1]}x{cropped.shape[0]})")

                try:
                    crop_queue.put_nowait(cropped)
                except queue.Full:
                    try:
                        crop_queue.get_nowait()
                        crop_queue.put_nowait(cropped)
                    except queue.Empty:
                        pass

                if args.debug:
                    save_crop_for_debug(cropped, config.DEBUG_DIR, crop_count)

                if args.max_crops > 0 and crop_count >= args.max_crops:
                    break

    except Exception as e:
        print(f"[Main] 실행 중 오류 발생: {e}")
    finally:
        print("[Main] 리소스 정리 중...")
        receiver.stop()
        inference_worker.stop()
        upload_worker.stop()
        
        receiver.join(timeout=2.0)
        inference_worker.join(timeout=2.0)
        upload_worker.join(timeout=2.0)

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n[Main] === 최종 통계 ===")
        print(f"  처리 프레임: {frame_count}, 포착 PCB: {crop_count}")
        print(f"  실행 시간: {elapsed:.1f}초, 평균 성능: {fps:.1f} FPS")


if __name__ == "__main__":
    main()
