"""
PCB 전처리 파이프라인

프레임에서 PCB를 감지하고 크롭하는 클래스
"""

import cv2
import numpy as np
from typing import Optional


class PCBPreprocessor:
    """
    PCB 전처리기

    상태 머신 기반으로 PCB 진입을 감지하고,
    배경 빼기로 PCB 영역을 크롭합니다.
    """

    def __init__(self, background_path: str):
        """
        Args:
            background_path: 배경 이미지 경로
        """
        self.background_gray = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        if self.background_gray is None:
            raise FileNotFoundError(f"배경 이미지를 찾을 수 없습니다: {background_path}")

        # 상태: "background" (PCB 없음) | "pcb" (PCB 진입)
        self.state = "background"
        self.crop_done = False

        # 검사 영역 (세로 띠) - 프레임 중앙 부근
        self.ROI_X1, self.ROI_X2 = 950, 970
        self.ROI_Y1, self.ROI_Y2 = 158, 922

        # 히스테리시스 임계값 (표준편차 기준)
        self.THRESH_LOW = 15   # 이 이하면 background로 전환
        self.THRESH_HIGH = 25  # 이 이상이면 pcb로 전환

        # 배경 빼기 임계값
        self.BG_DIFF_THRESH = 25

        # PCB 크기 범위 (영상 기준)
        self.MIN_HEIGHT, self.MAX_HEIGHT = 750, 780
        self.MIN_WIDTH, self.MAX_WIDTH = 700, 1600
        self.MIN_RATIO, self.MAX_RATIO = 0.8, 2.0  # 실제 PCB 비율 ~1.9

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        프레임 처리 후 크롭된 PCB 반환

        Args:
            frame: BGR 프레임 (1920x1080)

        Returns:
            크롭된 PCB 이미지 또는 None
        """
        # 1. 검사 영역 추출
        roi = frame[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]

        # 2. 표준편차 측정 (BGR - 구분력 더 높음)
        std = np.std(roi)

        # 3. 상태 판단 (히스테리시스)
        prev_state = self.state

        if self.state == "background" and std > self.THRESH_HIGH:
            self.state = "pcb"
        elif self.state == "pcb" and std < self.THRESH_LOW:
            self.state = "background"
            self._reset()

        # 4. 트리거 판단 (background → pcb 전환 시)
        trigger = (prev_state == "background" and self.state == "pcb")

        # 5. 크롭 시도 (PCB 상태이고 아직 크롭 안 했으면)
        if self.state == "pcb" and not self.crop_done:
            cropped = self._try_crop(frame)
            if cropped is not None:
                self.crop_done = True
                return cropped

        return None

    def _try_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        배경 빼기로 PCB 영역을 찾고 크롭

        Args:
            frame: BGR 프레임

        Returns:
            크롭된 PCB 이미지 또는 None (크기 검증 실패 시)
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 배경 빼기
        diff = cv2.absdiff(gray, self.background_gray)
        _, binary = cv2.threshold(diff, self.BG_DIFF_THRESH, 255, cv2.THRESH_BINARY)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_h, frame_w = frame.shape[:2]
        margin = 10

        # 조건에 맞는 컨투어 찾기 (크기 + 경계 체크)
        valid_contour = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 크기 검증
            if not self._validate_size(w, h):
                continue

            # 경계 체크 (프레임 안에 완전히 들어와야 함)
            if x < margin or (x + w) > (frame_w - margin):
                continue

            # 조건 만족하는 컨투어 발견
            valid_contour = (x, y, w, h)
            break

        if valid_contour is None:
            return None

        x, y, w, h = valid_contour
        print(f"[Preprocessor] 크롭 성공! bbox: x={x}, w={w}, size={w}x{h}")
        return frame[y:y+h, x:x+w].copy()

    def _validate_size(self, w: int, h: int) -> bool:
        """
        PCB 크기 유효성 검증

        Args:
            w: 너비
            h: 높이

        Returns:
            유효하면 True
        """
        # 높이 범위 확인
        if not (self.MIN_HEIGHT <= h <= self.MAX_HEIGHT):
            return False

        # 너비 범위 확인
        if not (self.MIN_WIDTH <= w <= self.MAX_WIDTH):
            return False

        # 가로세로 비율 확인
        ratio = w / h
        if not (self.MIN_RATIO <= ratio <= self.MAX_RATIO):
            return False

        return True

    def _reset(self):
        """상태 초기화 (PCB가 나갔을 때)"""
        self.crop_done = False

    def get_state(self) -> str:
        """현재 상태 반환"""
        return self.state

    def is_crop_done(self) -> bool:
        """현재 PCB에 대한 크롭 완료 여부"""
        return self.crop_done
