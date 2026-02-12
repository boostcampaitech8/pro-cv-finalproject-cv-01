import os

# RTSP 및 백엔드 서버 설정
RTSP_URL = "rtsp://3.36.185.146:8554/pcb_stream"
API_URL = "http://3.35.182.98:8080/detect/"

# 모델 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
BACKGROUND_PATH = os.path.join(BASE_DIR, "background.png")

# 전처리 파라미터 (ROI 및 임계값)
# 해상도 1920x1080 기준
ROI_X1, ROI_X2 = 950, 970
ROI_Y1, ROI_Y2 = 158, 922

THRESH_LOW = 15
THRESH_HIGH = 25
BG_DIFF_THRESH = 25

# PCB 크기 검증 파라미터
MIN_HEIGHT, MAX_HEIGHT = 750, 780
MIN_WIDTH, MAX_WIDTH = 700, 1600
MIN_RATIO, MAX_RATIO = 0.8, 2.0

# 시스템 설정
FRAME_QUEUE_SIZE = 2
CROP_QUEUE_SIZE = 10
UPLOAD_QUEUE_SIZE = 20

# 저장 경로
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
FAILED_DIR = os.path.join(STORAGE_DIR, "failed")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_crops")
