from fastapi import APIRouter, Query
from schemas.schemas import StatsResponse
from database import db
from typing import List, Dict, Optional

router = APIRouter(
    tags=["Stats"]
)


@router.get("/stats", response_model=StatsResponse)
async def get_statistics(session_id: Optional[int] = Query(None, description="세션 ID (없으면 전체)")):
    """
    Returns current statistics: total count, defect count, rate, etc.
    session_id가 제공되면 해당 세션의 통계만 반환.
    """
    return await db.get_stats(session_id=session_id)


@router.get("/latest", response_model=List[Dict])
async def get_latest_logs(
    limit: int = 10,
    session_id: Optional[int] = Query(None, description="세션 ID (없으면 전체)")
):
    """
    Returns the N most recent inspection logs.
    session_id가 제공되면 해당 세션의 로그만 반환.
    """
    return await db.get_recent_logs(limit, session_id=session_id)



@router.get("/defects", response_model=Dict[str, int])
async def get_defect_aggregation(
    session_id: Optional[int] = Query(None, description="세션 ID (없으면 전체)")
):
    """
    Returns aggregation of defect types.
    Example: {"scratch": 5, "dent": 2}
    session_id가 제공되면 해당 세션의 결함만 집계.
    """
    defect_logs = await db.get_defect_logs(session_id=session_id)

    aggregation = {}
    for log in defect_logs:
        # 새 스키마: detections 리스트에서 defect_type 추출
        detections = log.get("detections", [])
        if isinstance(detections, list):
            for det in detections:
                d_type = det.get("defect_type") if isinstance(det, dict) else None
                if d_type:
                    aggregation[d_type] = aggregation.get(d_type, 0) + 1
        else:
            # 기존 스키마 호환: defect_type 필드가 직접 있는 경우
            d_type = log.get("defect_type")
            if d_type:
                aggregation[d_type] = aggregation.get(d_type, 0) + 1
            else:
                aggregation["unknown"] = aggregation.get("unknown", 0) + 1

    return aggregation
