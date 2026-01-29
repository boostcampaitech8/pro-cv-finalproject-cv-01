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
import requests

# 상위 디렉토리의 rtsp 모듈 import를 위한 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rtsp'))

from preprocessor import PCBPreprocessor
from rtsp_receiver import RTSPReceiver
from inference_worker import InferenceWorker


# 기본 설정
DEFAULT_RTSP_URL = "rtsp://3.36.185.146:8554/pcb_stream"
DEFAULT_API_URL = "http://3.35.182.98:8080/detect/"
DEFAULT_SESSION_URL = "http://3.35.182.98:8080/sessions/"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
BACKGROUND_PATH = os.path.join(os.path.dirname(__file__), "background.png")
FRAME_QUEUE_SIZE = 2
CROP_QUEUE_SIZE = 10


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
        # sys.exit(1) # 테스트를 위해 일단 진행 가능하게 주석 처리하거나 에러 무시


def save_crop_for_debug(crop, output_dir: str, index: int):
    """디버그용 크롭 이미지 저장"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crop_{index:04d}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, crop)
    print(f"[Debug] 크롭 저장: {filepath}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="PCB 전처리 및 추론 파이프라인")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_RTSP_URL,
        help=f"RTSP URL 또는 비디오 파일 경로 (기본값: {DEFAULT_RTSP_URL})"
    )
    parser.add_argument(
        "--api-url", "-a",
        default=DEFAULT_API_URL,
        help=f"백엔드 API 주소 (기본값: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL_PATH,
        help=f"YOLO 모델 경로 (기본값: {DEFAULT_MODEL_PATH})"
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
    parser.add_argument(
        "--session-url",
        default=DEFAULT_SESSION_URL,
        help=f"세션 API 주소 (기본값: {DEFAULT_SESSION_URL})"
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="세션 관리 비활성화"
    )
    args = parser.parse_args()

    # 배경 이미지 확인
    if not os.path.exists(BACKGROUND_PATH):
        print(f"[Main] 경고: 배경 이미지({BACKGROUND_PATH})가 없습니다. 전처리 로직이 정상 작동하지 않을 수 있습니다.")

    # 세션 시작
    session_id = None
    if not args.no_session:
        session_id = start_session(args.session_url)

    # Queue 생성
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    crop_queue = queue.Queue(maxsize=CROP_QUEUE_SIZE)

    # 1. 추론 워커 먼저 초기화 (모델 로드 및 엔진 변환 수행)
    print(f"[Main] 추론 워커 초기화 중 (모델: {args.model})...")
    try:
        inference_worker = InferenceWorker(crop_queue, args.model, args.api_url, session_id=session_id)
    except Exception as e:
        print(f"[Main] 추론 워커 환경 설정 실패: {e}")
        # 세션 종료 후 종료
        if not args.no_session and session_id:
            end_session(args.session_url, session_id)
        sys.exit(1)

    # 2. RTSP 수신 스레드 시작 (모델 준비 완료 후)
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
        print("[Main] RTSP 연결 타임아웃 (무시하고 진행하거나 확인 필요)")

    # 3. 모든 준비가 끝나면 추론 워커 스레드 가동
    inference_worker.start()

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

    print("[Main] 파이프라인 가동 시작")
    print(f"  - 입력: {args.input}")
    print(f"  - API: {args.api_url}")
    print(f"  - 모델: {args.model}")
    print(f"  - 세션 ID: {session_id if session_id else '없음'}")

    crop_count = 0
    frame_count = 0
    start_time = time.time()

    try:
        while not shutdown_flag:
            try:
                # 프레임 가져오기
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not receiver.is_running():
                    print("[Main] 수신 스레드 종료됨")
                    break
                continue

            frame_count += 1

            # 전처리 수행 (PCB 감지 시 크롭 이미지 반환)
            cropped = preprocessor.process_frame(frame)

            if cropped is not None:
                crop_count += 1
                print(f"[Main] [#{crop_count}] PCB 포착! (크기: {cropped.shape[1]}x{cropped.shape[0]})")

                # crop_queue에 추가 (inference_worker가 가져가서 추론 및 전송)
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
        print(f"[Main] 실행 중 오류 발생: {e}")
    finally:
        # 정리
        print("[Main] 리소스 정리 중...")
        receiver.stop()
        inference_worker.stop()

        receiver.join(timeout=2.0)
        inference_worker.join(timeout=2.0)

        # 세션 종료
        if not args.no_session and session_id:
            end_session(args.session_url, session_id)

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n[Main] === 최종 통계 ===")
        print(f"  처리 프레임: {frame_count}")
        print(f"  포착된 PCB: {crop_count}")
        print(f"  실행 시간: {elapsed:.1f}초")
        print(f"  평균 성능: {fps:.1f} FPS")
        print(f"  수신 상태: {receiver.get_stats()}")
        print(f"  세션 ID: {session_id if session_id else '없음'}")



if __name__ == "__main__":
    main()
