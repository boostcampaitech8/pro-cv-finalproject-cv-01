import base64
import json
import logging
import traceback
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from config import settings
from database import db
from schemas.schemas import DetectRequest, DetectResponse
from utils import image_utils
from utils.auth import verify_api_key
from utils.slack_notifier import send_slack_alert

logger = logging.getLogger("uvicorn")

# Track previous health status per session.
last_status_per_session: dict = {}

router = APIRouter(prefix="/detect", tags=["Detect"])


def _parse_optional_int(raw_value) -> Optional[int]:
    if raw_value is None:
        return None
    raw = str(raw_value).strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid session_id: {raw_value}",
        ) from exc


def _parse_detections(raw_value) -> list:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    raw = str(raw_value).strip()
    if raw == "":
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid detections JSON in multipart payload",
        ) from exc
    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="detections must be a JSON array",
        )
    return parsed


async def _parse_request_payload(http_request: Request) -> Tuple[DetectRequest, Optional[bytes]]:
    content_type = (http_request.headers.get("content-type") or "").lower()

    if "multipart/form-data" in content_type:
        form = await http_request.form()
        req_dict = {
            "timestamp": form.get("timestamp"),
            "image_id": form.get("image_id"),
            "camera_id": form.get("camera_id"),
            "detections": _parse_detections(form.get("detections")),
            "session_id": _parse_optional_int(form.get("session_id")),
        }
        try:
            parsed_request = DetectRequest.model_validate(req_dict)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=exc.errors(),
            ) from exc

        image_file = form.get("image_file")
        image_bytes = None
        if image_file is not None and hasattr(image_file, "read"):
            image_bytes = await image_file.read()
            if hasattr(image_file, "close"):
                await image_file.close()
        return parsed_request, image_bytes

    try:
        body = await http_request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        ) from exc

    try:
        parsed_request = DetectRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc

    image_bytes = None
    if parsed_request.image:
        try:
            image_bytes = image_utils.decode_base64_image(parsed_request.image)
        except (base64.binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Base64 image data: {str(exc)}",
            ) from exc

    return parsed_request, image_bytes


@router.post(
    "/",
    response_model=DetectResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def receive_detection_result(http_request: Request):
    """
    Receives detection result from edge.

    Supported request formats:
    - application/json with optional base64 `image`
    - multipart/form-data with optional `image_file`
    """
    saved_image_path = None

    try:
        request, image_bytes = await _parse_request_payload(http_request)

        if image_bytes:
            try:
                saved_image_path = image_utils.save_image_to_s3(
                    image_bytes, request.image_id, request.timestamp
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to save image to S3: {str(exc)}",
                ) from exc

        log_ids = await db.add_inspection_log(request, saved_image_path)

        if settings.SLACK_ALERT_ENABLED:
            await check_and_send_slack_alert(request.session_id)

        return DetectResponse(status="ok", id=log_ids[0])

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(exc)}",
        ) from exc


async def check_and_send_slack_alert(session_id: Optional[int]):
    """Send Slack alert only when health status transitions."""
    try:
        session_filter = str(session_id) if session_id is not None else None
        health_data = await db.get_health(session_filter)
        current_status = health_data.status

        tracking_key = session_id if session_id is not None else "__global__"
        previous_status = last_status_per_session.get(tracking_key)

        if previous_status == current_status:
            logger.debug(
                f"[Slack] status unchanged (session={session_id}, status={current_status})"
            )
            return

        logger.info(
            f"[Slack] status changed: {previous_status or 'None'} -> {current_status} "
            f"(session={session_id})"
        )

        alerts_dict = [alert.model_dump() for alert in health_data.alerts]
        session_dict = health_data.session_info.model_dump()
        status_change_message = _get_status_change_message(previous_status, current_status)

        await send_slack_alert(
            status=current_status,
            alerts=alerts_dict,
            session_info=session_dict,
            status_change_message=status_change_message,
        )

        last_status_per_session[tracking_key] = current_status

    except Exception as exc:
        logger.error(f"[Slack] alert send failed: {exc}")


def _get_status_change_message(previous: Optional[str], current: str) -> str:
    if previous is None:
        return f"New session started. Current status: {current}"

    transitions = {
        ("healthy", "warning"): "Warning: status changed from healthy to warning.",
        ("healthy", "critical"): "Critical: status changed from healthy to critical.",
        ("warning", "critical"): "Escalation: status changed from warning to critical.",
        ("warning", "healthy"): "Recovered: status changed from warning to healthy.",
        ("critical", "warning"): "Partially recovered: critical to warning.",
        ("critical", "healthy"): "Fully recovered: critical to healthy.",
    }

    return transitions.get((previous, current), f"Status changed: {previous} -> {current}")
