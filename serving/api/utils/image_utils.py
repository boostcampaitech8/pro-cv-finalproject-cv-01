import base64
import boto3
import os
from config.settings import S3_BUCKET_NAME, S3_KEY_PREFIX, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)


def decode_base64_image(base64_string: str) -> bytes:
    """
    Decodes a Base64 string into bytes.
    Removes data URI header (e.g., 'data:image/jpeg;base64,') if present.
    """
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    return base64.b64decode(base64_string)


def save_image_to_s3(image_data: bytes, image_id: str, timestamp: str) -> str:
    """
    Uploads image bytes to AWS S3.
    Returns the S3 Key (path) where the image is stored.
    Format: raw/YYYYMMDD/{timestamp}_{image_id}.jpg
    """
    # Create Date based folder structure (YYYYMMDD from timestamp)
    # timestamp ex: 2026-01-14T15:30:45
    date_folder = timestamp.split("T")[0].replace("-", "") # 20260114
    
    # Sanitize timestamp for filename
    sanitized_timestamp = timestamp.replace(":", "-")
    filename = f"{sanitized_timestamp}_{image_id}.jpg"
    
    # Full S3 Key
    s3_key = f"{S3_KEY_PREFIX}{date_folder}/{filename}"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_data,
            ContentType="image/jpeg"
        )
        return s3_key
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        raise e


def generate_presigned_url(s3_key: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL to share an S3 object (GET).
    expiration: Time in seconds for the presigned URL to remain valid
    """
    try:
        response = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return response
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
