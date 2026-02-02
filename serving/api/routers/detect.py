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

            # Save to S3
            try:
                saved_image_path = image_utils.save_image_to_s3(
                    image_bytes,
                    request.image_id,
                    request.timestamp
                )
            except Exception as e:
                # S3 Upload Failed
                print(f"S3 Upload Error: {e}")
                # Optional: Fail gracefully or raise error. 
                # Here we raise error to alert the edge device.
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to save image to S3: {str(e)}"
                )

        # 2. Save Log(s) to Database
        log_ids = await db.add_inspection_log(request, saved_image_path)

        # 3. [신규] health 체크 및 Slack 알림 (비동기)
        if settings.SLACK_ALERT_ENABLED:
            await check_and_send_slack_alert(request.session_id)

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


async def check_and_send_slack_alert(session_id: Optional[int]):
    """
    health 체크 후 상태 변화 시에만 Slack 전송
    상태 변화 감지: healthy ↔ warning ↔ critical
    """
    try:
        # 세션 ID가 없으면 종료 (상태 추적 불가)
        if not session_id:
            logger.debug("[Slack] session_id 없음 - 상태 변화 감지 불가")
            return

        # health 데이터 조회
        health_data = await db.get_health(str(session_id))
        current_status = health_data.status

        # 이전 상태 조회
        previous_status = last_status_per_session.get(session_id)

        # 상태 변화 체크
        if previous_status == current_status:
            # 상태 변화 없음 → 알림 전송 안 함
            logger.debug(f"[Slack] 상태 유지 (세션 {session_id}, 상태: {current_status})")
            return

        # 상태 변화 감지 → Slack 전송
        logger.info(f"[Slack] 상태 변화 감지: {previous_status or 'None'} → {current_status} (세션 {session_id})")

        # Slack 메시지 생성
        alerts_dict = [alert.model_dump() for alert in health_data.alerts]
        session_dict = health_data.session_info.model_dump()

        # 상태 변화 메시지 추가
        status_change_message = _get_status_change_message(previous_status, current_status)

        await send_slack_alert(
            status=current_status,
            alerts=alerts_dict,
            session_info=session_dict,
            status_change_message=status_change_message
        )

        # 현재 상태 저장
        last_status_per_session[session_id] = current_status

    except Exception as e:
        logger.error(f"[Slack] 알림 전송 실패: {e}")


def _get_status_change_message(previous: Optional[str], current: str) -> str:
    """상태 변화에 따른 메시지 생성"""
    if previous is None:
        return f"🔔 세션 시작: 현재 상태 {current}"

    transitions = {
        ("healthy", "warning"): "⚠️ 경고: 시스템 상태가 warning으로 전환되었습니다",
        ("healthy", "critical"): "🚨 긴급: 시스템 상태가 critical로 전환되었습니다",
        ("warning", "critical"): "🚨 악화: 상태가 warning에서 critical로 악화되었습니다",
        ("warning", "healthy"): "✅ 개선: 시스템이 정상 상태로 복구되었습니다",
        ("critical", "warning"): "⚠️ 부분 개선: critical에서 warning으로 개선되었습니다",
        ("critical", "healthy"): "✅ 해결: 시스템이 완전히 복구되었습니다",
    }

    return transitions.get((previous, current), f"ℹ️ 상태 변화: {previous} → {current}")
