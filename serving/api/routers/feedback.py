from fastapi import APIRouter, HTTPException, status
from schemas.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    ImageStats,
    BboxStats,
    DefectTypeAccuracy,
    FeedbackTypeStats,
    ClassConfusion,
    FeedbackQueueResponse,
    RelabelRequest,
    BulkFeedbackRequest,
    BulkFeedbackResponse
)
from database import db
from utils import s3_dataset
from utils.image_utils import generate_presigned_url
from config.settings import get_class_id
from datetime import datetime
import logging
import json
from typing import List, Dict, Optional

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/feedback", tags=["Feedback"])


# ===== Helper Functions =====

def bbox_equals(bbox1: List[int], bbox2: List[int], tolerance: int = 2) -> bool:
    """
    두 bbox가 동일한지 확인 (오차 허용)

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        tolerance: 허용 오차 (픽셀)

    Returns:
        동일 여부
    """
    if not bbox1 or not bbox2:
        return False
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(bbox1, bbox2))


def find_feedback_by_bbox(
    feedbacks: List[Dict],
    bbox: List[int]
) -> Optional[Dict]:
    """
    target_bbox로 피드백 찾기 (좌표 매칭)

    Args:
        feedbacks: 피드백 목록 (dict)
        bbox: 원본 bbox [x1, y1, x2, y2]

    Returns:
        매칭된 피드백 또는 None
    """
    for feedback in feedbacks:
        if "target_bbox" in feedback and feedback["target_bbox"]:
            if bbox_equals(feedback["target_bbox"], bbox, tolerance=2):
                return feedback
    return None


def normalize_bbox(bbox_px: List[int], width: int, height: int) -> List[float]:
    """
    픽셀 좌표 → 정규화 좌표 (YOLO 형식)

    Args:
        bbox_px: [x1, y1, x2, y2] 픽셀 좌표
        width: 이미지 너비
        height: 이미지 높이

    Returns:
        [x_center, y_center, w, h] 정규화된 좌표 (0.0~1.0)
    """
    x1, y1, x2, y2 = bbox_px
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return [x_center, y_center, w, h]


# ===== API Endpoints =====

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
            created_by=request.created_by,
            target_bbox=request.target_bbox
        )

        logger.info(
            f"[피드백] log_id={request.log_id}, "
            f"type={request.feedback_type}, "
            f"target_bbox={request.target_bbox}, "
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


@router.post("/bulk", response_model=BulkFeedbackResponse, status_code=status.HTTP_201_CREATED)
async def create_bulk_feedback(request: BulkFeedbackRequest):
    """
    다중 bbox 피드백 생성 + 자동 S3 저장 (재라벨링 승인 스킵)

    1개 PCB(log_id)에 대해 여러 bbox에 대한 피드백을 제출하고,
    즉시 S3 refined/ 폴더에 학습 데이터를 저장합니다.

    플로우:
    1. inspection_logs 조회 (원본 detections)
    2. 각 bbox에 대해 피드백 찾기 (target_bbox 매칭)
    3. 최종 라벨 생성
       - 피드백 없음 → 원본 유지 (암묵적 TP)
       - tp_wrong_class → 클래스 수정
       - false_positive → 삭제
    4. S3 refined/ 저장 (이미지 복사 + 라벨 생성)
    5. DB 피드백 저장 (피드백 있는 bbox만)
    6. FALSE_NEGATIVE 처리 (있으면)
    7. inspection_logs.is_verified = true

    Example:
        POST /feedback/bulk
        {
          "log_id": 42,
          "image_width": 1200,
          "image_height": 760,
          "feedbacks": [
            {
              "target_bbox": [50, 60, 70, 80],
              "feedback_type": "false_positive",
              "comment": "먼지 오탐"
            },
            {
              "target_bbox": [100, 110, 120, 130],
              "feedback_type": "tp_wrong_class",
              "correct_label": "scratch",
              "comment": "hole이 아니라 scratch"
            }
          ],
          "false_negative_memo": "좌측 하단 scratch 누락",
          "created_by": "qa_team"
        }

    Args:
        request: 다중 피드백 요청 데이터

    Returns:
        생성된 피드백 ID 목록, S3 저장 정보, 최종 라벨 개수

    Raises:
        HTTPException 404: 존재하지 않는 log_id
        HTTPException 500: S3 저장 실패, DB 트랜잭션 실패
    """
    try:
        # 1. 원본 데이터 조회
        log = await db.get_inspection_log(request.log_id)
        if not log:
            raise HTTPException(status_code=404, detail=f"Inspection log {request.log_id} not found")

        original_detections = json.loads(log["detections"]) if log.get("detections") else []
        image_s3_key = log.get("image_path")  # "raw/20260203/xxx.jpg"

        if not image_s3_key:
            raise HTTPException(status_code=400, detail=f"Log {request.log_id} has no image_path")

        # 2. 피드백 데이터 변환
        feedbacks_data = [item.model_dump() for item in request.feedbacks]

        # 3. 최종 라벨 생성
        final_labels = []
        feedback_ids = []

        for detection in original_detections:
            bbox = detection["bbox"]

            # target_bbox로 피드백 찾기
            feedback = find_feedback_by_bbox(feedbacks_data, bbox)

            if feedback is None:
                # 피드백 없음 → 원본 그대로 유지 (암묵적 TP)
                final_labels.append({
                    "class_id": get_class_id(detection["defect_type"]),
                    "bbox": normalize_bbox(bbox, request.image_width, request.image_height)
                })
                # DB 저장 안 함

            elif feedback["feedback_type"] == "false_positive":
                # FP → 라벨에서 제외 (삭제)
                # DB에는 저장 (피드백 기록)
                pass  # final_labels에 추가 안 함 = 삭제

            elif feedback["feedback_type"] == "tp_wrong_class":
                # 클래스만 수정
                final_labels.append({
                    "class_id": get_class_id(feedback["correct_label"]),
                    "bbox": normalize_bbox(bbox, request.image_width, request.image_height)
                })

        # 4. S3 refined/ 저장
        refined_path = None
        saved_to_s3 = False

        if final_labels or len(original_detections) > 0:
            try:
                refined_path = await s3_dataset.save_to_refined(
                    image_s3_key=image_s3_key,
                    labels=final_labels,
                    log_id=request.log_id
                )
                saved_to_s3 = True
            except Exception as e:
                logger.error(f"[S3] Failed to save refined data for log_id={request.log_id}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save to S3: {str(e)}"
                )

        # 5. DB 피드백 저장 (피드백 있는 bbox만)
        for feedback_item in feedbacks_data:
            if feedback_item["feedback_type"] in ["false_positive", "tp_wrong_class"]:
                feedback_data = await db.add_feedback(
                    log_id=request.log_id,
                    feedback_type=feedback_item["feedback_type"],
                    target_bbox=feedback_item.get("target_bbox"),
                    correct_label=feedback_item.get("correct_label"),
                    comment=feedback_item.get("comment"),
                    created_by=request.created_by
                )
                feedback_ids.append(feedback_data["id"])

        # 6. FALSE_NEGATIVE 처리
        fn_pending = False
        if request.false_negative_memo:
            # DB에 저장
            fn_data = await db.add_feedback(
                log_id=request.log_id,
                feedback_type="false_negative",
                comment=request.false_negative_memo,
                created_by=request.created_by
            )
            feedback_ids.append(fn_data["id"])

            # needs_labeling/에 복사
            try:
                await s3_dataset.copy_to_needs_labeling(
                    image_s3_key=image_s3_key,
                    log_id=request.log_id,
                    memo=request.false_negative_memo,
                    original_detections=original_detections
                )
                fn_pending = True
            except Exception as e:
                logger.error(f"[S3] Failed to copy to needs_labeling for log_id={request.log_id}: {e}")
                # FALSE_NEGATIVE는 실패해도 계속 진행 (DB에는 저장됨)

        # 7. 검증 완료 표시
        await db.mark_as_verified(request.log_id, request.created_by)

        # 8. 로깅
        logger.info(
            f"[Bulk 피드백 + S3] log_id={request.log_id}, "
            f"feedback_count={len(feedback_ids)}, "
            f"final_labels={len(final_labels)}, "
            f"saved_to_s3={saved_to_s3}, "
            f"fn_pending={fn_pending}, "
            f"by={request.created_by or 'anonymous'}"
        )

        # 9. 응답 반환
        return BulkFeedbackResponse(
            status="ok",
            log_id=request.log_id,
            feedback_ids=feedback_ids,
            saved_to_s3=saved_to_s3,
            refined_path=refined_path,
            final_label_count=len(final_labels),
            false_negative_pending=fn_pending,
            created_at=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Bulk 피드백 실패] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create bulk feedback: {str(e)}"
        )


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_statistics(session_id: Optional[str] = None):
    """
    피드백 통계 조회 (bbox 기반 정확도 분석)

    MLOps 모니터링용:
    - image_stats: 전체 이미지 + 검증 진행률
    - bbox_stats: 검증된 defect의 bbox별 정확도 (핵심)
    - feedback_stats: 피드백 타입별 집계
    - class_confusion: 클래스 혼동 패턴

    Query Parameters:
        session_id: 세션 필터 ("latest", 숫자 ID, 생략 시 전체)
            - 생략: 전체 데이터 통계
            - "latest": 현재 진행 중인 세션의 통계
            - 숫자 (예: "1"): 해당 세션의 통계

    Returns:
        FeedbackStatsResponse: bbox 기반 정확도 분석 결과

    Raises:
        HTTPException 404: 세션을 찾을 수 없음
        HTTPException 500: 서버 에러
    """
    try:
        # 세션 필터 처리 (기존 resolve_session_filter 재사용)
        resolved_session_id = await db.resolve_session_filter(session_id)
        stats_data = await db.get_feedback_stats(session_id=resolved_session_id)

        # image_stats 변환
        image_stats = ImageStats(**stats_data["image_stats"])

        # bbox_stats 변환 (by_defect_type 내 DefectTypeAccuracy 변환)
        bbox_data = stats_data["bbox_stats"]
        by_defect_type = {
            dt: DefectTypeAccuracy(**dt_data)
            for dt, dt_data in bbox_data["by_defect_type"].items()
        }
        bbox_stats = BboxStats(
            total=bbox_data["total"],
            correct=bbox_data["correct"],
            false_positive=bbox_data["false_positive"],
            wrong_class=bbox_data["wrong_class"],
            accuracy_rate=bbox_data["accuracy_rate"],
            by_defect_type=by_defect_type
        )

        # feedback_stats 변환
        feedback_stats = FeedbackTypeStats(**stats_data["feedback_stats"])

        # class_confusion 변환
        class_confusion = [
            ClassConfusion(**item)
            for item in stats_data["class_confusion"]
        ]

        return FeedbackStatsResponse(
            image_stats=image_stats,
            bbox_stats=bbox_stats,
            feedback_stats=feedback_stats,
            class_confusion=class_confusion
        )

    except Exception as e:
        logger.error(f"[피드백 통계 실패] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feedback stats: {str(e)}"
        )

@router.get("/queue", response_model=List[FeedbackQueueResponse])
async def get_labeling_queue(session_id: Optional[str] = None):
    """
    라벨링 대기열 조회
    처리되지 않은 피드백 목록을 반환합니다.

    Query Parameters:
        session_id: 세션 필터 ("latest", 숫자 ID, 생략 시 전체)
            - 생략: 전체 대기열
            - "latest": 현재 진행 중인 세션의 대기열
            - 숫자 (예: "1"): 해당 세션의 대기열
    """
    try:
        # 세션 필터 처리
        resolved_session_id = await db.resolve_session_filter(session_id)
        queue_items = await db.get_feedback_queue(session_id=resolved_session_id)
        
        response_list = []
        for item in queue_items:
            # 1. Image URL 생성 (Presigned URL)
            image_url = generate_presigned_url(item["image_path"])

            # 2. Detections JSON 파싱
            try:
                detections = json.loads(item["detections"])
            except:
                detections = []

            # 3. target_bbox JSON 파싱
            try:
                target_bbox = json.loads(item["target_bbox"]) if item.get("target_bbox") else None
            except:
                target_bbox = None

            response_list.append(FeedbackQueueResponse(
                feedback_id=item["feedback_id"],
                log_id=item["log_id"],
                image_url=image_url,
                feedback_type=item["feedback_type"],
                comment=item["comment"],
                created_at=item["created_at"],
                original_detections=detections,
                target_bbox=target_bbox
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
