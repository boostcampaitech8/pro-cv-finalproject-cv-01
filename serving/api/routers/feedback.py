from fastapi import APIRouter, HTTPException, status
from schemas.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    FeedbackTypeStats,
    DefectTypeFeedbackStats
)
from database import db
import logging

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
