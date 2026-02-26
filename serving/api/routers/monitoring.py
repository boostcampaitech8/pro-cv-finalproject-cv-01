from fastapi import APIRouter, Query
from typing import Optional
from database import db
from schemas.schemas import HealthResponse, AlertsResponse

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health", response_model=HealthResponse)
async def get_health_status(
    session_id: Optional[str] = Query(None, description="세션 ID ('latest', 숫자, 또는 생략)")
):
    """
    시스템 건강 상태 모니터링 API

    - session_id="latest": 진행 중인 세션
    - session_id=숫자: 특정 세션
    - session_id 생략: 전체 데이터

    **알림 임계값:**
    - 불량률: warning 10%, critical 20%
    - 평균 신뢰도: warning 0.85, critical 0.75
    - 저신뢰도 비율: warning 20%, critical 40%
    - PCB당 평균 결함: warning 2.0개, critical 3.0개
    """
    return await db.get_health(session_id)


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    session_id: Optional[str] = Query(None, description="세션 ID ('latest', 숫자, 또는 생략)")
):
    """
    알림 조회 API (프론트엔드 폴링용 경량 버전)

    - /monitoring/health의 alerts 필드만 반환
    - 프론트엔드 실시간 폴링에 최적화

    **사용 예시:**
    - GET /monitoring/alerts (전체 데이터)
    - GET /monitoring/alerts?session_id=latest (최신 세션)
    - GET /monitoring/alerts?session_id=1 (특정 세션)
    """
    health_data = await db.get_health(session_id)

    # HealthResponse Pydantic 모델에서 필요한 필드 추출
    avg_confidence = 0.0
    if health_data.defect_confidence_stats:
        avg_confidence = health_data.defect_confidence_stats.avg_confidence

    return {
        "status": health_data.status,
        "timestamp": health_data.timestamp,
        "session_info": health_data.session_info,
        "alerts": health_data.alerts,
        "summary": {
            "defect_rate": health_data.defect_rate,
            "avg_confidence": avg_confidence
        },
        "active_mlops_version": health_data.active_mlops_version,
        "active_yolo_version": health_data.active_yolo_version
    }
