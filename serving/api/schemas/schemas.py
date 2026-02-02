from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict

# --- Shared Models ---

class DefectInfo(BaseModel):
    """
    Model representing detailed defect information.
    Used in StatsResponse(last_defect) and potentially for internal logic.
    """
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    image_id: str = Field(..., description="Image Identifier")
    result: str = Field(..., description="'defect' or 'normal'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    defect_type: Optional[str] = Field(None, description="Type of defect used if result is defect")
    bbox: Optional[List[int]] = Field(None, description="[x1, y1, x2, y2] bounding box")
    image_path: Optional[str] = Field(None, description="Path to the saved image file")


class Detection(BaseModel):
    """
    개별 결함 정보
    """
    defect_type: str = Field(..., description="결함 종류 (scratch, hole, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] bounding box")


# --- Request Models ---

class DetectRequest(BaseModel):
    """
    Request body for POST /detect
    여러 결함을 지원하는 새로운 구조
    """
    timestamp: str = Field(..., description="ISO 8601 timestamp, e.g., '2026-01-14T15:30:45'")
    image_id: str = Field(..., description="Image Identifier, e.g., 'PCB_001234'")
    image: Optional[str] = Field(None, description="Base64 encoded image string (optional)")
    detections: List[Detection] = Field(default_factory=list, description="결함 목록 (빈 배열 = 정상)")
    session_id: Optional[int] = Field(None, description="세션 ID (optional)")


# --- Response Models ---


# --- Response Models ---

class DetectResponse(BaseModel):
    """
    Response for POST /detect
    """
    status: str = "ok"
    id: int


class InspectionLogResponse(BaseModel):
    """
    Standard response model for an inspection log entry.
    Matches the new DB structure (1 row per image).
    """
    id: int
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    image_id: str = Field(..., description="Image Identifier")
    result: str = Field(..., description="'defect' or 'normal'")
    detections: List[Detection] = Field(default_factory=list, description="List of detected defects")
    image_path: Optional[str] = Field(None, description="Path to saved image file")
    session_id: Optional[int] = Field(None, description="세션 ID")
    # 기존 confidence, defect_type, bbox 필드는 detections 리스트 안에 포함됨


class StatsResponse(BaseModel):
    """
    Response for GET /stats
    """
    total_inspections: int = Field(..., description="총 검사 수 (PCB 개수)")
    normal_count: int = Field(..., description="정상 PCB 개수")
    defect_items: int = Field(..., description="불량 PCB 개수")
    total_defects: int = Field(..., description="탐지된 결함 총 개수")
    defect_rate: float = Field(..., description="불량률 (%)")
    avg_defects_per_item: float = Field(..., description="불량 PCB당 평균 결함 개수")
    avg_fps: float = Field(..., description="Average processing/transmission FPS")
    last_defect: Optional[InspectionLogResponse] = Field(None, description="Most recent defect log entry")


# ===== 세션 관련 스키마 =====

class SessionResponse(BaseModel):
    """
    세션 정보 응답
    """
    id: int = Field(..., description="세션 ID")
    started_at: str = Field(..., description="세션 시작 시간 (ISO 8601)")
    ended_at: Optional[str] = Field(None, description="세션 종료 시간 (ISO 8601)")


class SessionCreateResponse(BaseModel):
    """
    세션 생성 응답
    """
    id: int = Field(..., description="생성된 세션 ID")
    started_at: str = Field(..., description="세션 시작 시간 (ISO 8601)")


class SessionListResponse(BaseModel):
    """
    세션 목록 응답
    """
    sessions: List[SessionResponse] = Field(default_factory=list, description="세션 목록")


# ===== Health Monitoring 관련 스키마 =====

class AlertInfo(BaseModel):
    """
    알림 정보
    """
    level: str = Field(..., description="알림 레벨 (warning, critical)")
    message: str = Field(..., description="알림 메시지")
    value: float = Field(..., description="현재 값")
    threshold: float = Field(..., description="임계값")
    action: str = Field(..., description="권장 조치")


class ConfidenceDistribution(BaseModel):
    """
    신뢰도 분포
    """
    high: int = Field(..., description="높은 신뢰도 (0.9-1.0)")
    medium: int = Field(..., description="중간 신뢰도 (0.8-0.9)")
    low: int = Field(..., description="낮은 신뢰도 (0.7-0.8)")
    very_low: int = Field(..., description="매우 낮은 신뢰도 (<0.7)")


class DefectConfidenceStats(BaseModel):
    """
    결함 신뢰도 통계
    """
    avg_confidence: float = Field(..., description="평균 신뢰도")
    min_confidence: float = Field(..., description="최소 신뢰도")
    max_confidence: float = Field(..., description="최대 신뢰도")
    distribution: ConfidenceDistribution = Field(..., description="신뢰도 분포")


class DefectTypeStat(BaseModel):
    """
    결함 타입별 통계
    """
    defect_type: str = Field(..., description="결함 종류")
    count: int = Field(..., description="발생 횟수")
    avg_confidence: float = Field(..., description="평균 신뢰도")


class SessionInfo(BaseModel):
    """
    세션 정보 (Health API용)
    """
    id: Optional[int] = Field(None, description="세션 ID")
    started_at: Optional[str] = Field(None, description="시작 시간")
    ended_at: Optional[str] = Field(None, description="종료 시간")
    duration_seconds: Optional[float] = Field(None, description="진행 시간 (초)")
    is_active: bool = Field(..., description="진행 중 여부")


class HealthResponse(BaseModel):
    """
    시스템 건강 상태 응답
    """
    status: str = Field(..., description="시스템 상태 (healthy, warning, critical)")
    timestamp: str = Field(..., description="응답 시각 (ISO 8601)")
    session_info: SessionInfo = Field(..., description="세션 정보")
    total_inspections: int = Field(..., description="총 검사 수")
    normal_count: int = Field(..., description="정상 개수")
    defect_count: int = Field(..., description="불량 개수")
    defect_rate: float = Field(..., description="불량률 (%)")
    total_defects: int = Field(..., description="총 결함 개수")
    avg_defects_per_item: float = Field(..., description="불량 PCB당 평균 결함 개수")
    defect_confidence_stats: Optional[DefectConfidenceStats] = Field(None, description="결함 신뢰도 통계")
    defect_type_stats: List[DefectTypeStat] = Field(default_factory=list, description="결함 타입별 통계")
    alerts: List[AlertInfo] = Field(default_factory=list, description="알림 목록")


class AlertsResponse(BaseModel):
    """
    알림 조회 응답 (Health API의 경량 버전)
    프론트엔드 폴링용 최적화
    """
    status: str = Field(..., description="시스템 상태 (healthy, warning, critical)")
    timestamp: str = Field(..., description="응답 시각 (ISO 8601)")
    session_info: SessionInfo = Field(..., description="세션 정보")
    alerts: List[AlertInfo] = Field(default_factory=list, description="알림 목록")
    summary: dict = Field(..., description="간단한 요약 (defect_rate, avg_confidence)")


# ===== 피드백 관련 스키마 =====

class FeedbackRequest(BaseModel):
    """
    피드백 생성 요청
    """
    log_id: int = Field(..., gt=0, description="검사 로그 ID")
    feedback_type: str = Field(..., description="피드백 종류 (false_positive, false_negative, label_correction)")
    correct_label: Optional[str] = Field(None, description="올바른 라벨 (label_correction 시 필수)")
    comment: Optional[str] = Field(None, max_length=500, description="추가 설명")
    created_by: Optional[str] = Field(None, max_length=100, description="작성자")

    @field_validator('feedback_type')
    @classmethod
    def validate_feedback_type(cls, v: str) -> str:
        allowed = {'false_positive', 'false_negative', 'label_correction'}
        if v not in allowed:
            raise ValueError(f"feedback_type must be one of {allowed}")
        return v

    @model_validator(mode='after')
    def validate_correct_label_requirement(self):
        from config.settings import ALLOWED_FEEDBACK_LABELS

        # label_correction 타입인 경우 correct_label 필수 및 검증
        if self.feedback_type == 'label_correction':
            if not self.correct_label or self.correct_label.strip() == '':
                raise ValueError("correct_label required for label_correction")
            # 허용값 검증
            if self.correct_label not in ALLOWED_FEEDBACK_LABELS:
                raise ValueError(
                    f"correct_label must be one of {ALLOWED_FEEDBACK_LABELS}, got '{self.correct_label}'"
                )
        return self


class FeedbackResponse(BaseModel):
    """
    피드백 생성 응답
    """
    id: int = Field(..., description="피드백 ID")
    log_id: int = Field(..., description="검사 로그 ID")
    feedback_type: str = Field(..., description="피드백 종류")
    correct_label: Optional[str] = Field(None, description="올바른 라벨")
    comment: Optional[str] = Field(None, description="추가 설명")
    created_at: str = Field(..., description="생성 시각 (ISO 8601)")
    created_by: Optional[str] = Field(None, description="작성자")
    status: str = Field(default="ok", description="응답 상태")


class FeedbackTypeStats(BaseModel):
    """
    피드백 종류별 통계
    """
    false_positive: int = Field(default=0, description="오탐 개수 (정상인데 불량으로)")
    false_negative: int = Field(default=0, description="미탐 개수 (불량인데 정상으로)")
    label_correction: int = Field(default=0, description="라벨 수정 개수 (결함 종류 틀림)")


class DefectTypeFeedbackStats(BaseModel):
    """
    결함 타입별 피드백 통계
    """
    defect_type: str = Field(..., description="결함 종류")
    false_positive: int = Field(default=0, description="오탐 개수")
    false_negative: int = Field(default=0, description="미탐 개수")
    label_correction: int = Field(default=0, description="라벨 수정 개수")


class FeedbackStatsResponse(BaseModel):
    """
    피드백 통계 응답
    """
    total_feedback: int = Field(..., description="전체 피드백 개수")
    by_type: FeedbackTypeStats = Field(..., description="피드백 종류별 집계")
    by_defect_type: List[DefectTypeFeedbackStats] = Field(
        default_factory=list,
        description="결함 타입별 피드백 집계"
    )
    recent_feedback_count: int = Field(..., description="최근 24시간 피드백 개수")
    period_description: str = Field(default="최근 24시간", description="집계 기간")


class FeedbackQueueResponse(BaseModel):
    """
    라벨링 대기열 아이템
    """
    feedback_id: int = Field(..., description="피드백 ID")
    log_id: int = Field(..., description="로그 ID")
    image_url: str = Field(..., description="이미지 다운로드 URL")
    feedback_type: str = Field(..., description="신고 유형")
    comment: Optional[str] = Field(None, description="사용자 코멘트")
    created_at: str = Field(..., description="신고 시간")
    # 기존 AI 예측 정보 (참고용)
    original_detections: List[Dict] = Field(default_factory=list, description="기존 AI 예측 결과")


class RelabelRequest(BaseModel):
    """
    재라벨링 승인 요청 데이터
    """
    feedback_id: int = Field(..., description="대상 피드백 ID")
    final_class_id: int = Field(..., description="확정된 클래스 ID")
    final_bbox: List[float] = Field(..., description="확정된 좌표 [x_center, y_center, w, h] Normalized")
    
    # 좌표 정규화를 위해 필요할 수 있음 (선택)
    image_width: Optional[int] = Field(None, description="이미지 너비")
    image_height: Optional[int] = Field(None, description="이미지 높이")
