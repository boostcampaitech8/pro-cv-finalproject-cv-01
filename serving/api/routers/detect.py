from fastapi import APIRouter, HTTPException, status, Depends
from schemas.schemas import DetectRequest, DetectResponse
from database import db
from utils import image_utils
from utils.slack_notifier import send_slack_alert
from utils.auth import verify_api_key
from config import settings
from datetime import datetime
from typing import Optional
import base64
import traceback
import logging

# 로거 설정
logger = logging.getLogger("uvicorn")

# 세션별 마지막 상태 추적 (상태 변화 감지용)
last_status_per_session: dict[int, str] = {}

router = APIRouter(
    prefix="/detect",
    tags=["Detect"]
)

@router.post("/", response_model=DetectResponse, status_code=status.HTTP_200_OK, dependencies=[Depends(verify_api_key)])
async def receive_detection_result(request: DetectRequest):
    """
    Receives detection result from Edge/Jetson.
    Supports multiple defects per image.
    """
    saved_image_path = None

    try:
        # 1. Handle Image Saving (이미지가 있으면 저장)
        if request.image:
            # Decode Base64
            try:
                image_bytes = image_utils.decode_base64_image(request.image)
            except (base64.binascii.Error, ValueError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid Base64 image data: {str(e)}"
                )

            # Save to disk
            try:
                saved_image_path = image_utils.save_defect_image(
                    image_bytes,
                    request.image_id,
                    request.timestamp
                )
            except IOError as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to save image: {str(e)}"
                )

        # 2. Save Log(s) to Database
        log_ids = await db.add_inspection_log(request, saved_image_path)

        return DetectResponse(status="ok", id=log_ids[0])  # 첫번째 ID 반환

    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )
