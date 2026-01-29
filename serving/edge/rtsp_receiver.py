"""
RTSP 수신 스레드

RTSP 스트림 또는 비디오 파일에서 프레임을 읽어 Queue에 전달
"""

import os
import time
import cv2
import queue
import threading
from typing import Optional

# RTSP TCP 모드 설정 (UDP 패킷 손실 방지)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class RTSPReceiver(threading.Thread):
    """
    RTSP 수신 스레드

    백그라운드에서 프레임을 읽어 Queue에 넣습니다.
    Queue가 가득 차면 오래된 프레임을 버립니다.
    """

    def __init__(
        self,
        source: str,
        frame_queue: queue.Queue,
        loop: bool = False
    ):
        """
        Args:
            source: RTSP URL 또는 비디오 파일 경로
            frame_queue: 프레임을 넣을 Queue
            loop: 비디오 파일 반복 재생 여부 (테스트용)
        """
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue
        self.loop = loop
        self.running = False
        self._stop_event = threading.Event()

        # 통계
        self.frame_count = 0
        self.drop_count = 0

    def run(self):
        """스레드 메인 루프"""
        print(f"[RTSPReceiver] 연결 시도: {self.source}")

        # RTSP 연결 옵션 설정
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화 (실시간성)

        if not cap.isOpened():
            print(f"[RTSPReceiver] 소스를 열 수 없습니다: {self.source}")
            return

        # 스트림 정보 출력
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[RTSPReceiver] 연결 성공: {width}x{height} @ {fps:.1f}fps")

        self.running = True
        print(f"[RTSPReceiver] 수신 시작...")

        consecutive_failures = 0
        max_failures = 30  # 연속 실패 허용 횟수 (약 1초)

        while not self._stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    if self.loop:
                        # 비디오 파일 반복 재생
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        consecutive_failures = 0
                        continue
                    else:
                        print(f"[RTSPReceiver] 스트림 종료 (연속 {consecutive_failures}회 실패)")
                        break
                time.sleep(0.033)  # ~30fps 대기
                continue

            consecutive_failures = 0  # 성공 시 리셋

            self.frame_count += 1

            # Queue가 가득 차면 오래된 프레임 버림 (실시간성 우선)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.drop_count += 1
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.drop_count += 1

        cap.release()
        self.running = False
        print(f"[RTSPReceiver] 종료 - 총 프레임: {self.frame_count}, 드롭: {self.drop_count}")

    def stop(self):
        """스레드 정지 요청"""
        self._stop_event.set()

    def is_running(self) -> bool:
        """실행 중 여부"""
        return self.running

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            "frame_count": self.frame_count,
            "drop_count": self.drop_count,
            "running": self.running
        }
