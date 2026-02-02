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
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
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
ALLOWED_DEFECT_TYPES = {"scratch", "hole", "contamination", "crack"}
ALLOWED_FEEDBACK_LABELS = ALLOWED_DEFECT_TYPES | {"normal"}  # 피드백에서는 normal도 허용
