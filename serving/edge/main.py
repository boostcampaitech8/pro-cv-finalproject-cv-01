"""
PCB 전처리 파이프라인 메인 모듈

RTSP 영상을 수신하고, PCB를 감지/크롭하여 추론 및 업로드 수행
"""

import argparse
import os
# Jetson: TensorRT는 시스템 패키지이므로 pip 자동 설치 방지
# os.environ["YOLO_AUTOINSTALL"] = "false"
import queue
import signal
import sys
import time
import subprocess
from datetime import datetime

import cv2
import requests

import config
from preprocessor import PCBPreprocessor
from rtsp_receiver import RTSPReceiver
from inference_worker import InferenceWorker
from upload_worker import UploadWorker




def get_current_model_version() -> dict:
    """
    현재 모델 버전 정보를 반환.
    current_version.json 파일이 있으면 해당 딕셔너리를, 없으면 기본값 반환.
    """
    version_file = os.path.join(os.path.dirname(config.MODEL_PATH), "current_version.json")
    
    # 기본값
    v_info = {
        "model_name": "yolov11m_v0",
        "yolo_version": "yolov11m",
        "mlops_version": "v0"
    }
    
    if os.path.exists(version_file):
        try:
            import json
            with open(version_file, 'r') as f:
                data = json.load(f)
                print(f"[Main] 모델 버전 파일 로드: {version_file}")
                # updater.py가 쓰는 model_name 필드 처리
                if "model_name" in data:
                    v_info["model_name"] = data["model_name"]
                    print(f"       - 로드된 모델명: {v_info['model_name']}")
                    # model_name에서 yolo_version과 mlops_version 파싱 시도 (선택 사항)
                    parts = data["model_name"].split('_')
                    if len(parts) >= 2:
                        v_info["yolo_version"] = parts[0]
                        v_info["mlops_version"] = parts[1]
                else:
                    v_info.update(data)
        except Exception as e:
            print(f"[Main] 모델 버전 파일 읽기 에러: {e}")
            
    return v_info


def start_session(session_url: str, model_info: dict) -> int:
    """
    백엔드에 세션 시작을 요청하고 세션 ID를 반환.
    실패 시 None 반환.
    """
    try:
        api_key = os.getenv("EDGE_API_KEY")
        headers = {}
        if api_key:
            headers["X-API-KEY"] = api_key
            
        model_name = model_info.get("model_name", "yolov11m_v0")
        print(f"[Main] 세션 요청 시작 (모델: {model_name})")
        payload = {
            "model_name": model_name
        }
        response = requests.post(session_url, json=payload, headers=headers, timeout=5.0)
        if response.status_code == 201:
            data = response.json()
            session_id = data.get("id")
            print(f"[Main] 🚀 세션 시작 성공!")
            print(f"       - 세션 ID: {session_id}")
            print(f"       - 활성 모델: {model_name}")
            return session_id
        else:
            print(f"[Main] ❌ 세션 시작 실패: HTTP {response.status_code} ({response.text})")
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
        api_key = os.getenv("EDGE_API_KEY")
        headers = {}
        if api_key:
            headers["X-API-KEY"] = api_key
            
        response = requests.patch(f"{session_url}{session_id}", headers=headers, timeout=5.0)
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
        "--no-updater",
        action="store_true",
        help="업데이트 프로세스(updater.py) 자동 실행 비활성화"
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
    
    args = parser.parse_args()

    # 입력 소스 처리
    # num_cameras > 1일 때 기본 RTSP 주소 뒤에 "_1", "_2" ...을 붙입니다.
    # 단, 비디오 파일(.mp4/.avi/.mkv)인 경우에는 같은 파일을 복제합니다.
    # num_cameras == 1이면 사용자가 전달한 입력 문자열을 그대로 사용하여
    # 특정 주소를 직접 지정할 수 있도록 합니다.
    if args.num_cameras >= 1:
        base_url = args.input
        if args.num_cameras == 1:
            # 단일 카메라는 suffix 없이 입력 그대로
            input_sources = [base_url]
        else:
            # 다중 카메라일 때만 _1, _2 ... 붙이기
            if not any(base_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv']):
                input_sources = [f"{base_url}_{i+1}" for i in range(args.num_cameras)]
            else:
                input_sources = [base_url] * args.num_cameras
    else:
        # num_cameras 가 0 이면 comma-separated list를 그대로 사용
        input_sources = [s.strip() for s in args.input.split(',')]

    # 동적 설정 업데이트 (CLI 인자 우선)
    config.API_URL = args.api_url
    config.MODEL_PATH = args.model
    
    # 세션 URL이 기본값인 경우 API_URL을 기준으로 자동 생성 (포트/경로 일관성 유지)
    session_url = args.session_url
    if "3.35.182.98" in session_url and "3.35.182.98" not in config.API_URL:
        from urllib.parse import urlparse
        parsed = urlparse(config.API_URL)
        session_url = f"{parsed.scheme}://{parsed.netloc}/sessions/"
        print(f"[Main] 세션 API 주소 자동 감지: {session_url}")

    # 배경 이미지 확인
    if not os.path.exists(config.BACKGROUND_PATH):
        print(f"[Main] 에러: 배경 이미지({config.BACKGROUND_PATH})가 없습니다.")
        sys.exit(1)

    # 세션 시작
    session_id = None
    if not args.no_session:
        model_info = get_current_model_version()
        session_id = start_session(session_url, model_info=model_info)

    # Queue 생성
    frame_queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE * len(input_sources))
    crop_queue = queue.Queue(maxsize=config.CROP_QUEUE_SIZE)
    upload_queue = queue.Queue(maxsize=config.UPLOAD_QUEUE_SIZE)

    # 1. 업로드 워커 시작
    upload_worker = UploadWorker(upload_queue)
    upload_worker.start()

    # 2. 추론 워커 초기화
    print(f"[Main] 추론 워커 초기화 중 (모델: {config.MODEL_PATH})...")
    try:
        inference_worker = InferenceWorker(crop_queue, upload_queue, config.MODEL_PATH, 
                                           session_id=session_id, session_url=args.session_url if not args.no_session else None)
    except Exception as e:
        print(f"[Main] 추론 워커 초기화 실패: {e}")
        if session_id:
            end_session(args.session_url, session_id)
        sys.exit(1)

    # 3. RTSP 수신 스레드 리스트 시작
    receivers = []
    for i, source in enumerate(input_sources):
        camera_id = f"cam_{i+1}"
        receiver = RTSPReceiver(source, frame_queue, camera_id=camera_id, loop=args.loop)
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

    # 5. 백그라운드 업데이터 프로세스는 main이 직접 실행하지 않습니다.
    #    외부에서 updater.py를 별도 서비스/스크립트로 구동하세요.
    updater_process = None

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
    print(f"  - 입력 소스: {len(input_sources)}개")
    for r in receivers:
        print(f"    * [{r.camera_id}] {r.source}")
    print(f"  - API: {config.API_URL}")
    print(f"  - 모델: {config.MODEL_PATH}")
    print(f"  - 세션 ID: {session_id if session_id else '없음'}")

    crop_count = 0
    frame_count = 0
    start_time = time.time()

    try:
        while not shutdown_flag:
            try:
                # 큐에서 (camera_id, frame) 튜플을 가져옴
                camera_id, frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if all(not r.is_running() for r in receivers):
                    # 모든 수신기가 오프라인이더라도 메인 프로세스는 종료하지 않고
                    # 재접속을 기다립니다. (RTSPReceiver가 재접속을 시도함)
                    time.sleep(1)
                    continue
                continue

            frame_count += 1
            cropped = preprocessor.process_frame(frame)

            if cropped is not None:
                crop_count += 1
                print(f"[Main][{camera_id}] [#{crop_count}] PCB 포착! ({cropped.shape[1]}x{cropped.shape[0]})")

                try:
                    crop_queue.put_nowait((camera_id, cropped))
                except queue.Full:
                    try:
                        crop_queue.get_nowait()
                        crop_queue.put_nowait((camera_id, cropped))
                    except queue.Empty:
                        pass

                if args.debug:
                    save_crop_for_debug(cropped, config.DEBUG_DIR, crop_count)

                if args.max_crops > 0 and crop_count >= args.max_crops:
                    break

    except Exception as e:
        print(f"[Main] 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Main] 리소스 정리 중...")
        
        # 모든 워커 정지
        for r in receivers:
            r.stop()
        inference_worker.stop()
        upload_worker.stop()

        for r in receivers:
            r.join(timeout=2.0)
        inference_worker.join(timeout=2.0)
        upload_worker.join(timeout=2.0)

        if session_id:
            end_session(args.session_url, session_id)

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n[Main] === 최종 통계 ===")
        print(f"  처리 프레임: {frame_count}, 포착 PCB: {crop_count}, 실행 시간: {elapsed:.1f}s, 평균 FPS: {fps:.1f}")
        if receivers:
            stats = ", ".join(f"{r.camera_id}:{r.get_stats()['frame_count']}f/{r.get_stats()['drop_count']}d" for r in receivers)
            print(f"  카메라 통계 ({len(receivers)}): {stats}")
        print(f"  세션 ID: {session_id if session_id else '없음'}")


if __name__ == "__main__":
    main()
