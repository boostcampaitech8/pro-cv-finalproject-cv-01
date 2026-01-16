from fastapi import APIRouter, HTTPException, status
from schemas.schemas import DetectRequest, DetectResponse
from database import db
from utils import image_utils
import base64
import traceback

router = APIRouter(
    prefix="/detect",
    tags=["Detect"]
)

@router.post("/", response_model=DetectResponse, status_code=status.HTTP_200_OK)
async def receive_detection_result(request: DetectRequest):
    """
    Receives detection result from Edge/Jetson.
    If result is 'defect' and image data is provided, saves the image.
    Saves the log to database (memory/sqlite).
    """
    saved_image_path = None

    try:
        # 1. Handle Defect Image Saving
        if request.result == "defect" and request.image:
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

        # 2. Save Log to Database
        log_id = await db.add_inspection_log(request, saved_image_path)

        return DetectResponse(status="ok", id=log_id)

    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )
