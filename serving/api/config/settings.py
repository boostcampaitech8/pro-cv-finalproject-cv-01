import os
from pathlib import Path

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
