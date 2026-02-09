from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from utils.image_utils import generate_presigned_url, S3_KEY_PREFIX

router = APIRouter(
    prefix="/raw",
    tags=["Images"]
)


@router.get("/{file_path:path}")
async def get_raw_image(file_path: str):
    """
    Redirects request to S3 Presigned URL.
    Frontend calls: /raw/20260114/timestamp_id.jpg
    Backend redirects to: https://s3.aws.com/...?signature=...
    """
    # Construct full S3 key
    # If file_path is '20260114/file.jpg', full key should be 'raw/20260114/file.jpg'
    full_s3_key = f"{S3_KEY_PREFIX}{file_path}"

    presigned_url = generate_presigned_url(full_s3_key)

    if presigned_url:
        return RedirectResponse(url=presigned_url, status_code=307)
    else:
        # 404 Not Found returns a JSON error, or could raise HTTPException
        return {"error": "Could not generate URL or file not found"}
