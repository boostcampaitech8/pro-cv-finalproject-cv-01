import boto3
import logging
import json
from datetime import datetime
from config import settings
from botocore.exceptions import ClientError
from typing import List, Dict, Optional

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


async def save_to_refined(
    image_s3_key: str,
    labels: List[Dict],
    log_id: int
) -> str:
    """
    refined/ 폴더에 이미지 복사 + YOLO 라벨 생성

    Args:
        image_s3_key: raw/20260203/xxx.jpg
        labels: [{"class_id": 0, "bbox": [x, y, w, h]}, ...]
        log_id: 검사 로그 ID

    Returns:
        refined_path: refined/images/42_xxx.jpg
    """
    bucket = settings.S3_BUCKET_NAME

    # 파일명 생성: {log_id}_{원본파일명}
    original_filename = image_s3_key.split("/")[-1]  # xxx.jpg
    refined_filename = f"{log_id}_{original_filename}"

    # 1. 이미지 복사 (S3 Copy Object)
    refined_image_key = f"refined/images/{refined_filename}"
    try:
        s3_client.copy_object(
            CopySource={'Bucket': bucket, 'Key': image_s3_key},
            Bucket=bucket,
            Key=refined_image_key
        )
    except ClientError as e:
        logger.error(f"[S3] Failed to copy image: {image_s3_key} -> {refined_image_key}, error: {e}")
        raise

    # 2. 라벨 파일 생성 (YOLO 포맷)
    label_content = ""
    for label in labels:
        # class_id x_center y_center width height (정규화된 값)
        label_content += f"{label['class_id']} {label['bbox'][0]:.6f} {label['bbox'][1]:.6f} {label['bbox'][2]:.6f} {label['bbox'][3]:.6f}\n"

    refined_label_key = f"refined/labels/{refined_filename.replace('.jpg', '.txt')}"
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=refined_label_key,
            Body=label_content,
            ContentType="text/plain"
        )
    except ClientError as e:
        logger.error(f"[S3] Failed to save label: {refined_label_key}, error: {e}")
        raise

    logger.info(f"[S3] Refined data saved: {refined_image_key}, {len(labels)} labels")
    return refined_image_key


async def copy_to_needs_labeling(
    image_s3_key: str,
    log_id: int,
    memo: str,
    original_detections: Optional[List[Dict]] = None
) -> None:
    """
    FALSE_NEGATIVE 이미지를 needs_labeling/ 폴더에 복사

    Args:
        image_s3_key: raw/20260203/xxx.jpg
        log_id: 검사 로그 ID
        memo: FALSE_NEGATIVE 메모
        original_detections: 원본 AI 예측 결과 (선택)
    """
    bucket = settings.S3_BUCKET_NAME
    original_filename = image_s3_key.split("/")[-1]
    needs_labeling_filename = f"{log_id}_{original_filename}"

    # 1. 이미지 복사
    needs_labeling_image_key = f"needs_labeling/images/{needs_labeling_filename}"
    try:
        s3_client.copy_object(
            CopySource={'Bucket': bucket, 'Key': image_s3_key},
            Bucket=bucket,
            Key=needs_labeling_image_key
        )
    except ClientError as e:
        logger.error(f"[S3] Failed to copy image to needs_labeling: {image_s3_key}, error: {e}")
        raise

    # 2. 메타데이터 JSON 생성
    metadata = {
        "log_id": log_id,
        "original_s3_key": image_s3_key,
        "false_negative_memo": memo,
        "original_detections": original_detections or [],
        "created_at": datetime.now().isoformat()
    }

    metadata_key = f"needs_labeling/metadata/{needs_labeling_filename.replace('.jpg', '.json')}"
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=json.dumps(metadata, ensure_ascii=False, indent=2),
            ContentType="application/json"
        )
    except ClientError as e:
        logger.error(f"[S3] Failed to save metadata to needs_labeling: {metadata_key}, error: {e}")
        raise

    logger.info(f"[S3] FALSE_NEGATIVE saved: {needs_labeling_image_key}")
