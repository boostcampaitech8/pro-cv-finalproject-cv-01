"""
RTSP 스트림에서 배경 이미지 캡처

가운데 배경 스트립을 캡처해서 가로로 타일링하여 전체 배경 생성
"""

import os
import cv2
import numpy as np

# RTSP TCP 모드 설정
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

RTSP_URL = "rtsp://3.36.185.146:8554/pcb_stream"
OUTPUT_PATH = "background.png"

# 프레임 크기
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# 캡처할 스트립 영역 (가운데 세로 띠)
STRIP_X1, STRIP_X2 = 955, 965  # 10px 너비
STRIP_Y1, STRIP_Y2 = 0, FRAME_HEIGHT  # 전체 높이

# 배경 판단 임계값 (표준편차가 이 이하면 배경으로 간주)
BACKGROUND_THRESH = 15


def create_background_from_strip(strip: np.ndarray) -> np.ndarray:
    """
    세로 스트립을 가로로 타일링해서 전체 배경 생성

    Args:
        strip: (높이, 너비, 3) 형태의 세로 스트립

    Returns:
        (1080, 1920, 3) 크기의 배경 이미지
    """
    strip_width = strip.shape[1]

    # 필요한 반복 횟수 계산
    num_tiles = (FRAME_WIDTH // strip_width) + 1

    # 가로로 타일링
    tiled = np.tile(strip, (1, num_tiles, 1))

    # 정확한 너비로 자르기
    background = tiled[:, :FRAME_WIDTH, :]

    return background


def main():
    print(f"[Capture] RTSP 연결: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[Capture] 연결 실패")
        return

    print("[Capture] 연결 성공, 배경 스트립 대기 중...")
    print(f"[Capture] 가운데 영역(x={STRIP_X1}~{STRIP_X2})의 표준편차가 {BACKGROUND_THRESH} 이하인 프레임을 찾습니다.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # 가운데 스트립 추출
        strip = frame[STRIP_Y1:STRIP_Y2, STRIP_X1:STRIP_X2]
        std = np.std(strip)

        if frame_count % 30 == 0:  # 1초마다 상태 출력
            print(f"[Capture] 프레임 #{frame_count}, 스트립 표준편차: {std:.1f}")

        # 배경 스트립 감지 (PCB 없음)
        if std < BACKGROUND_THRESH:
            print(f"\n[Capture] 배경 스트립 감지! 표준편차: {std:.1f}")

            # 스트립을 타일링해서 전체 배경 생성
            background = create_background_from_strip(strip)

            cv2.imwrite(OUTPUT_PATH, background)
            print(f"[Capture] 배경 생성 완료!")
            print(f"[Capture] 스트립 크기: {strip.shape[1]}x{strip.shape[0]}")
            print(f"[Capture] 배경 크기: {background.shape[1]}x{background.shape[0]}")
            print(f"[Capture] 저장: {OUTPUT_PATH}")
            break

    cap.release()
    print("[Capture] 종료")


if __name__ == "__main__":
    main()
