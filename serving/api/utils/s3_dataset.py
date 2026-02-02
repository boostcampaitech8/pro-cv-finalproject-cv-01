import boto3
import logging
from config import settings
from botocore.exceptions import ClientError
from typing import List, Dict

logger = logging.getLogger("uvicorn")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

async def save_refined_data(
    original_s3_key: str,
    class_id: int,
    bbox: List[float],
    filename_stem: str
) -> bool:
    """
    승인된 데이터를 S3 refined 폴더에 저장합니다.
    1. 원본 이미지 Copy: raw/... -> refined/images/...
    2. 라벨 파일 생성: refined/labels/....txt
    
    Args:
        original_s3_key: S3상의 원본 이미지 경로 (예: raw/20260202/defect_123.jpg)
        class_id: 확정된 클래스 ID
        bbox: [x_center, y_center, w, h] (0.0 ~ 1.0 Normalized)
        filename_stem: 저장할 파일명 (확장자 제외, 예: defect_123_refined)
    """
    bucket = settings.S3_BUCKET_NAME
    
    try:
        # 1. 이미지 복사 (Copy Object)
        # 원본이 존재하는지 확인하지 않고 바로 Copy 시도 (없으면 에러)
        target_image_key = f"refined/images/{filename_stem}.jpg"
        
        copy_source = {
            'Bucket': bucket,
            'Key': original_s3_key
        }
        
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=bucket,
            Key=target_image_key
        )
        logger.info(f"[S3 Dataset] Image copied: {original_s3_key} -> {target_image_key}")

        # 2. 라벨 파일(.txt) 생성 및 업로드
        # YOLO Format: class x_center y_center width height
        label_content = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
        target_label_key = f"refined/labels/{filename_stem}.txt"
        
        s3_client.put_object(
            Bucket=bucket,
            Key=target_label_key,
            Body=label_content,
            ContentType="text/plain"
        )
        logger.info(f"[S3 Dataset] Label saved: {target_label_key}")
        
        return True

    except ClientError as e:
        logger.error(f"[S3 Dataset] Failed to save refined data: {e}")
        raise e
    except Exception as e:
        logger.error(f"[S3 Dataset] Unexpected error: {e}")
        raise e

async def get_refined_dataset_stats() -> dict:
    """
    S3 refined 폴더의 데이터 통계를 반환합니다.
    """
    bucket = settings.S3_BUCKET_NAME
    prefix = "refined/images/"
    
    try:
        # Paginator를 사용하여 전체 객체 수 카운트
        paginator = s3_client.get_paginator('list_objects_v2')
        count = 0
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                count += len(page['Contents'])
                
        return {
            "image_count": count,
            "s3_prefix": f"refined/",
            "bucket": bucket
        }
    except Exception as e:
        logger.error(f"[S3 Stats] Failed to get stats: {e}")
        return {"image_count": 0, "error": str(e)}
