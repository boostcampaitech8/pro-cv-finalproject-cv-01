"""
PCB 전처리 파이프라인 메인 모듈

RTSP 영상을 수신하고, PCB를 감지/크롭하여 추론 Queue에 전달
"""

import argparse
import os
import queue
import signal
import sys
import time
from datetime import datetime

import cv2

# 상위 디렉토리의 rtsp 모듈 import를 위한 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rtsp'))

from preprocessor import PCBPreprocessor
from rtsp_receiver import RTSPReceiver


# 기본 설정
DEFAULT_RTSP_URL = "rtsp://3.36.185.146:8554/pcb_stream"
BACKGROUND_PATH = os.path.join(os.path.dirname(__file__), "background.png")
FRAME_QUEUE_SIZE = 2
CROP_QUEUE_SIZE = 10


def generate_background_if_needed():
    """배경 이미지가 없으면 생성"""
    if os.path.exists(BACKGROUND_PATH):
        print(f"[Main] 배경 이미지 존재: {BACKGROUND_PATH}")
        return

    print("[Main] 배경 이미지 생성 중...")
    try:
        from pcb_video import generate_background
        generate_background(BACKGROUND_PATH)
    except ImportError:
        print("[Main] pcb_video 모듈을 찾을 수 없습니다. 배경 이미지를 수동으로 생성하세요.")
        sys.exit(1)


def save_crop_for_debug(crop: 'np.ndarray', output_dir: str, index: int):
    """디버그용 크롭 이미지 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crop_{index:04d}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, crop)
    print(f"[Debug] 크롭 저장: {filepath}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="PCB 전처리 파이프라인")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_RTSP_URL,
        help=f"RTSP URL 또는 비디오 파일 경로 (기본값: {DEFAULT_RTSP_URL})"
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
        "--debug-dir",
        default="debug_crops",
        help="디버그 크롭 저장 디렉토리 (기본값: debug_crops)"
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="최대 크롭 개수 (0=무제한, 테스트용)"
    )
    args = parser.parse_args()

    # 배경 이미지 확인/생성
    generate_background_if_needed()

    # Queue 생성
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    crop_queue = queue.Queue(maxsize=CROP_QUEUE_SIZE)

    # RTSP 수신 스레드 시작
    receiver = RTSPReceiver(args.input, frame_queue, loop=args.loop)
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
    else:
        print("[Main] RTSP 연결 타임아웃")
        receiver.stop()
        sys.exit(1)

    # 전처리기 초기화
    preprocessor = PCBPreprocessor(BACKGROUND_PATH)

    # 종료 시그널 핸들러
    shutdown_flag = False

    def signal_handler(signum, frame):
        nonlocal shutdown_flag
        print("\n[Main] 종료 신호 수신...")
        shutdown_flag = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("[Main] 전처리 파이프라인 시작")
    print(f"[Main] 입력 소스: {args.input}")
    print(f"[Main] 디버그 모드: {args.debug}")

    crop_count = 0
    frame_count = 0
    start_time = time.time()

    try:
        while not shutdown_flag:
            try:
                # 프레임 가져오기 (타임아웃으로 종료 체크 가능하게)
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not receiver.is_running():
                    print("[Main] 수신 스레드 종료됨")
                    break
                continue

            frame_count += 1

            # 전처리 수행
            cropped = preprocessor.process_frame(frame)

            if cropped is not None:
                crop_count += 1
                print(f"[Main] PCB 크롭 #{crop_count} - 크기: {cropped.shape[1]}x{cropped.shape[0]}")

                # crop_queue에 추가 (추론 워커용)
                try:
                    crop_queue.put_nowait(cropped)
                except queue.Full:
                    # Queue가 가득 차면 오래된 것 버림
                    try:
                        crop_queue.get_nowait()
                        crop_queue.put_nowait(cropped)
                    except queue.Empty:
                        pass

                # 디버그 모드: 크롭 이미지 저장
                if args.debug:
                    save_crop_for_debug(cropped, args.debug_dir, crop_count)

                # 최대 크롭 개수 도달 시 종료
                if args.max_crops > 0 and crop_count >= args.max_crops:
                    print(f"[Main] 최대 크롭 개수 도달: {args.max_crops}")
                    break

    except Exception as e:
        print(f"[Main] 오류 발생: {e}")
        raise
    finally:
        # 정리
        receiver.stop()
        receiver.join(timeout=2.0)

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n[Main] === 통계 ===")
        print(f"  처리 프레임: {frame_count}")
        print(f"  크롭 개수: {crop_count}")
        print(f"  실행 시간: {elapsed:.1f}초")
        print(f"  평균 FPS: {fps:.1f}")
        print(f"  수신 스레드 통계: {receiver.get_stats()}")


if __name__ == "__main__":
    main()
