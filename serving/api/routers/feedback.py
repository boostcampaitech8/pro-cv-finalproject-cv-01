from fastapi import APIRouter, HTTPException, status
from schemas.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    FeedbackTypeStats,
    DefectTypeFeedbackStats,
    FeedbackQueueResponse,
    RelabelRequest
)
from database import db
from utils import s3_dataset
from utils.image_utils import generate_presigned_url
import logging
import json
from typing import List, Dict

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("/", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def create_feedback(request: FeedbackRequest):
    """
    검사 결과에 대한 피드백 생성

    피드백 종류:
    - false_positive: 정상인데 불량으로 오탐
    - false_negative: 불량인데 정상으로 통과
    - label_correction: 결함은 맞지만 종류가 틀림

    Args:
        request: 피드백 요청 데이터

    Returns:
        생성된 피드백 정보

    Raises:
        HTTPException 404: 존재하지 않는 log_id
        HTTPException 422: 유효성 검증 실패
        HTTPException 500: 서버 에러
    """
    try:
        feedback_data = await db.add_feedback(
            log_id=request.log_id,
            feedback_type=request.feedback_type,
            correct_label=request.correct_label,
            comment=request.comment,
            created_by=request.created_by
        )

        logger.info(
            f"[피드백] log_id={request.log_id}, "
            f"type={request.feedback_type}, "
            f"by={request.created_by or 'anonymous'}"
        )

        return FeedbackResponse(**feedback_data, status="ok")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[피드백 실패] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create feedback: {str(e)}"
        )


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_statistics():
    """
    피드백 통계 조회

    MLOps 모니터링용:
    - 피드백 종류별 집계 (FP/FN/LC)
    - 결함 타입별 피드백 집계
    - 최근 24시간 피드백 개수

    Returns:
        피드백 통계 정보

    Raises:
        HTTPException 500: 서버 에러
    """
    try:
        stats_data = await db.get_feedback_stats()

        return FeedbackStatsResponse(
            total_feedback=stats_data["total_feedback"],
            by_type=FeedbackTypeStats(**stats_data["by_type"]),
            by_defect_type=[
                DefectTypeFeedbackStats(**item)
                for item in stats_data["by_defect_type"]
            ],
            recent_feedback_count=stats_data["recent_feedback_count"],
            period_description="최근 24시간"
        )

    except Exception as e:
        logger.error(f"[피드백 통계 실패] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feedback stats: {str(e)}"
        )

@router.get("/queue", response_model=List[FeedbackQueueResponse])
async def get_labeling_queue():
    """
    라벨링 대기열 조회
    처리되지 않은 피드백 목록을 반환합니다.
    """
    try:
        queue_items = await db.get_feedback_queue()
        
        response_list = []
        for item in queue_items:
            # 1. Image URL 생성 (Presigned URL)
            image_url = generate_presigned_url(item["image_path"])
            
            # 2. Detections JSON 파싱
            try:
                detections = json.loads(item["detections"])
            except:
                detections = []

            response_list.append(FeedbackQueueResponse(
                feedback_id=item["feedback_id"],
                log_id=item["log_id"],
                image_url=image_url,
                feedback_type=item["feedback_type"],
                comment=item["comment"],
                created_at=item["created_at"],
                original_detections=detections
            ))
            
        return response_list

    except Exception as e:
        logger.error(f"[대기열 조회 실패] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue: {str(e)}"
        )

@router.post("/approve")
async def approve_relabeling(request: RelabelRequest):
    """
    재라벨링 승인 및 저장
    1. S3 refined 폴더에 이미지와 라벨 저장
    2. DB 피드백 상태 완료 처리
    """
    try:
        # 1. 대상 로그 정보 가져오기 (이미지 경로 필요)
        # 큐에서 조회했던 정보 다시 조회하거나, log_id를 통해 조회
        # 여기서는 feedback_id로 조회하는 함수가 필요하지만, 
        # 간단히 queue 쿼리를 재활용하거나 새로 만듭니다.
        # 효율성을 위해 db.get_feedback_queue() 로직을 일부 활용하거나
        # get_log_by_feedback_id 같은 함수가 있으면 좋음.
        # 일단 여기서는 구현 편의상 로그 ID를 통해 이미지 경로를 찾는다고 가정하거나 
        # DB에 get_feedback_detail 함수를 추가하는 것이 정석입니다.
        
        # 임시: queue 목록에서 해당 ID 찾기 (비효율적이지만 안전)
        # 실제로는 db.get_feedback_by_id(request.feedback_id) 필요
        # db.py 수정 없이 가려면... 
        # -> 차라리 db.py에 get_image_path_by_feedback_id 추가하는게 나음.
        # 하지만 스텝 최소화를 위해 queue 쿼리에서 필터링
        
        queue_items = await db.get_feedback_queue()
        target_item = next((item for item in queue_items if item["feedback_id"] == request.feedback_id), None)
        
        if not target_item:
            raise HTTPException(status_code=404, detail="Feedback not found or already resolved")

        image_path = target_item["image_path"] # 예: raw/2026.../img.jpg
        filename_stem = image_path.split("/")[-1].replace(".jpg", "") + f"_refined_{request.feedback_id}"

        # 2. S3 저장 (Copy & Label)
        await s3_dataset.save_refined_data(
            original_s3_key=image_path,
            class_id=request.final_class_id,
            bbox=request.final_bbox,
            filename_stem=filename_stem
        )

        # 3. DB 상태 업데이트
        await db.resolve_feedback(request.feedback_id)

        logger.info(f"[재라벨링 승인] ID={request.feedback_id} -> S3 Refined Saved")
        return {"status": "ok", "message": "Relabeling approved and saved"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[승인 실패] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve: {str(e)}"
        )

@router.get("/export")
async def export_dataset():
    """
    MLOps 파이프라인 연동용 데이터셋 정보 반환
    학습 서버에서 이 API를 호출하여 데이터 위치와 개수를 확인합니다.
    """
    try:
        from config import settings
        
        # S3 데이터 통계 조회
        stats = await s3_dataset.get_refined_dataset_stats()
        
        s3_uri = f"s3://{settings.S3_BUCKET_NAME}/{stats['s3_prefix']}"
        
        return {
            "status": "ready",
            "dataset_info": {
                "s3_uri": s3_uri,
                "bucket": settings.S3_BUCKET_NAME,
                "prefix": stats["s3_prefix"],
                "image_count": stats["image_count"]
            },
            "command_guide": f"aws s3 sync {s3_uri} ./dataset/refined/",
            "message": "Use the s3_uri to sync your training data."
        }
    except Exception as e:
        logger.error(f"[Export] Failed to get dataset info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export dataset info: {str(e)}"
        )
