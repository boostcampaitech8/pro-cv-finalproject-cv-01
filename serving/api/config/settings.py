import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory (serving/api/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Database
DB_PATH = BASE_DIR / "data" / "inspection.db"

# Image storage
IMAGE_DIR = BASE_DIR / "images" / "defects"

# CORS settings
CORS_ORIGINS = ["*"]  # 프로덕션에서는 구체적인 도메인으로 변경
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# API settings
API_TITLE = "PCB Defect Detection API"
API_DESCRIPTION = "Backend for Jetson Edge AI and Streamlit Frontend"
API_VERSION = "1.0.0"

# Edge Device Authentication
EDGE_API_KEY = os.getenv("EDGE_API_KEY", None)
# AWS S3 Settings
# 환경 변수에서 로드하거나, .env 파일이 없으면 아래 기본값을 사용 (보안상 환경변수 권장)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "pcb-data-storage")
S3_KEY_PREFIX = "raw/"

# Health Monitoring Alert Thresholds
ALERT_THRESHOLDS = {
    # 불량률 임계값 (%)
    "defect_rate_warning": 10.0,
    "defect_rate_critical": 20.0,

    # 평균 신뢰도 임계값
    "avg_confidence_warning": 0.85,
    "avg_confidence_critical": 0.75,

    # 낮은 신뢰도 비율 임계값 (%)
    "low_confidence_ratio_warning": 20.0,
    "low_confidence_ratio_critical": 40.0,

    # 불량 PCB당 평균 결함 개수
    "avg_defects_per_item_warning": 2.0,
    "avg_defects_per_item_critical": 3.0,
}

# Slack Webhook Settings
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")  # 환경변수로 관리
SLACK_ALERT_ENABLED = os.getenv("SLACK_ALERT_ENABLED", "false").lower() == "true"

# Defect Type Settings
ALLOWED_DEFECT_TYPES = {"missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"}
ALLOWED_FEEDBACK_LABELS = ALLOWED_DEFECT_TYPES | {"normal"}  # 피드백에서는 normal도 허용

# YOLO 클래스 ID 매핑 (학습 데이터 생성용)
CLASS_ID_MAP = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5
}

def get_class_id(defect_type: str) -> int:
    """
    결함 타입을 YOLO 클래스 ID로 변환
    엣지에서 "Missing Hole" 형식으로 전송되므로 정규화 처리

    Args:
        defect_type: 결함 종류 ("Missing Hole", "missing_hole" 등)

    Returns:
        클래스 ID (0-based), 매칭 실패 시 -1
    """
    # 정규화: 소문자 + 공백을 언더스코어로 변환
    normalized = defect_type.strip().lower().replace(" ", "_")
    result = CLASS_ID_MAP.get(normalized, -1)
    if result == -1:
        import logging
        logging.getLogger("uvicorn").warning(
            f"[CLASS_ID] 알 수 없는 결함 타입: '{defect_type}' (정규화: '{normalized}')"
        )
    return result
