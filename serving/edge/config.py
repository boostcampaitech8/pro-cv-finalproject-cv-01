import os

# RTSP 및 백엔드 서버 설정
RTSP_URL = "rtsp://3.36.185.146:8554/pcb_stream"
API_URL = "http://3.35.182.98:8080/detect/"

# 모델 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "current.engine")
RELOAD_FLAG_PATH = os.path.join(MODELS_DIR, ".reload_model")
BUILDING_FLAG_PATH = os.path.join(MODELS_DIR, ".building_engine")
BACKGROUND_PATH = os.path.join(BASE_DIR, "background.png")
INFERENCE_BUSY_FLAG_PATH = os.path.join(MODELS_DIR, ".inference_busy")
RTSP_RETRY_INTERVAL_S = int(os.getenv("RTSP_RETRY_INTERVAL_S", "300"))
INFERENCE_IDLE_TIMEOUT_S = int(os.getenv("INFERENCE_IDLE_TIMEOUT_S", "5"))

# Golden Set 경로 (evaluate_engine용 고정 검증 데이터셋)
GOLDEN_SET_DIR = os.path.join(BASE_DIR, "golden_set")
GOLDEN_YAML_PATH = os.path.join(GOLDEN_SET_DIR, "golden.yaml")

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

# ── TRT 빌드 리소스 상한선 ────────────────────────────────────────────────
# Jetson Orin Nano Super: 8GB 통합 LPDDR5 (CPU/GPU 공유, VRAM 별도 없음)
#   - 추론 중 병행 빌드 시: 현재 엔진 ~1-2GB + OS ~1GB 사용 중
#   - YOLOv8 빌드 피크 메모리 ≈ 모델크기 × 4 (YOLOv8s ~30MB → ~120MB 피크)
#   - 1024MB: 여유 있는 안전한 상한 (2048MB 이상은 OOM 위험)
TRT_WORKSPACE_MB     = int(os.getenv("TRT_WORKSPACE_MB",     "1024"))
# minTiming=1: 최소 1회 측정 (빌드 속도 우선)
# avgTiming=8: 8회 평균 (엔진 품질과 빌드 시간의 균형, NVIDIA 권장 기본값)
TRT_MIN_TIMING_ITERS = int(os.getenv("TRT_MIN_TIMING_ITERS", "1"))
TRT_AVG_TIMING_ITERS = int(os.getenv("TRT_AVG_TIMING_ITERS", "8"))
# trtexec 빌드 최대 허용 시간 (초). 초과 시 실패 처리하여 무한 대기 방지
TRT_BUILD_TIMEOUT_S  = int(os.getenv("TRT_BUILD_TIMEOUT_S",  "600"))

# ── 모델 폴링 및 조건 필터 ───────────────────────────────────────────────
# S3 latest.json을 확인하는 주기 (초). 환경변수로 Jetson마다 개별 조정 가능
MODEL_POLL_INTERVAL  = int(os.getenv("MODEL_POLL_INTERVAL",  "300"))  # 기본 5분
# Jetson이 수락할 모델의 필수 태그 조건 (latest.json의 "tags" 딕셔너리 참조)
# 예: REQUIRED_TAG_KEY=status, REQUIRED_TAG_VALUE=retrained
#     → tags["status"] == "retrained" 인 모델만 채택
# 비워두면 (빈 문자열) 모든 모델 허용
REQUIRED_TAG_KEY     = os.getenv("REQUIRED_TAG_KEY",   "status")
REQUIRED_TAG_VALUE   = os.getenv("REQUIRED_TAG_VALUE", "retrained")
