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
    camera_id: Optional[str] = Field(None, description="카메라 식별자 (e.g., cam_1)")
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
    camera_id: Optional[str] = Field(None, description="카메라 식별자")
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
    high: int = Field(..., description="높은 신뢰도 (>=0.8)")
    mid: int = Field(..., description="중간 신뢰도 (0.5-0.8)")
    low: int = Field(..., description="낮은 신뢰도 (<0.5)")


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
    low_confidence_ratio: float = Field(default=0.0, description="저신뢰도 결함 비율 (%)")
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


# ===== 피드백 관련 스키마 (Bulk 전용) =====

class FeedbackItem(BaseModel):
    """
    개별 bbox 피드백 (bulk 요청의 단위)
    """
    feedback_type: str
    correct_label: Optional[str] = None
    comment: Optional[str] = Field(None, max_length=500)
    target_bbox: Optional[List[int]] = None

    @field_validator('feedback_type')
    @classmethod
    def validate_feedback_type(cls, v: str) -> str:
        allowed = {'false_positive', 'false_negative', 'tp_wrong_class'}
        if v not in allowed:
            raise ValueError(f"feedback_type must be one of {allowed}")
        return v

    @model_validator(mode='after')
    def validate_feedback_requirements(self):
        from config.settings import ALLOWED_FEEDBACK_LABELS

        # FeedbackRequest와 동일한 검증 로직 재사용 (DRY 원칙)
        if self.feedback_type == 'tp_wrong_class':
            if not self.correct_label:
                raise ValueError("correct_label required for tp_wrong_class")
            if self.correct_label not in ALLOWED_FEEDBACK_LABELS:
                raise ValueError(f"correct_label must be one of {ALLOWED_FEEDBACK_LABELS}")

        if self.feedback_type in ['false_positive', 'tp_wrong_class']:
            if not self.target_bbox:
                raise ValueError(f"target_bbox required for {self.feedback_type}")

        if self.target_bbox:
            if len(self.target_bbox) != 4:
                raise ValueError("target_bbox must have 4 elements [x1, y1, x2, y2]")
            if self.target_bbox[0] >= self.target_bbox[2]:
                raise ValueError("x1 must be less than x2")
            if self.target_bbox[1] >= self.target_bbox[3]:
                raise ValueError("y1 must be less than y2")

        return self


class BulkFeedbackRequest(BaseModel):
    """
    다중 bbox 피드백 생성 요청 (자동 재라벨링)

    false_negative는 feedbacks 배열에 포함하여 제출:
    {"feedback_type": "false_negative", "comment": "좌측 하단 scratch 누락"}
    """
    log_id: int = Field(..., gt=0)
    image_width: int = Field(..., gt=0, description="크롭된 이미지 너비 (픽셀)")
    image_height: int = Field(..., gt=0, description="크롭된 이미지 높이 (픽셀)")
    feedbacks: List[FeedbackItem] = Field(..., description="피드백 목록 (false_negative 포함)")
    created_by: Optional[str] = Field(None, max_length=100)

    @field_validator('feedbacks')
    @classmethod
    def validate_feedbacks_count(cls, v: List[FeedbackItem]) -> List[FeedbackItem]:
        # 빈 배열 허용 (모든 bbox가 정답일 수 있음)
        if len(v) > 100:
            raise ValueError(f"feedbacks cannot exceed 100 items, got {len(v)}")
        return v


class BulkFeedbackResponse(BaseModel):
    """
    다중 bbox 피드백 생성 응답 (자동 재라벨링)
    """
    status: str = Field(default="ok")
    log_id: int
    feedback_ids: List[int] = Field(..., description="생성된 피드백 ID 목록")
    saved_to_s3: bool = Field(..., description="S3 refined/ 폴더 저장 여부")
    refined_path: Optional[str] = Field(None, description="refined/ 이미지 경로")
    final_label_count: int = Field(..., description="최종 라벨 개수")
    false_negative_count: int = Field(..., description="FALSE_NEGATIVE 개수")
    created_at: str


# ===== Feedback Stats 응답 스키마 (bbox 기반 정확도 분석) =====

class ImageStats(BaseModel):
    """
    이미지 단위 통계 (전체 + 검증 진행률)
    """
    total: int = Field(..., description="전체 이미지 개수")
    by_result: Dict[str, int] = Field(..., description="결과별 개수 {'defect': 4, 'normal': 2}")
    verified: int = Field(..., description="검증 완료 이미지 개수")
    unverified: int = Field(..., description="미검증 이미지 개수")
    verification_rate: float = Field(..., description="검증률 (%)")
    verified_by_result: Dict[str, int] = Field(..., description="검증된 이미지의 결과별 개수")


class DefectTypeAccuracy(BaseModel):
    """
    결함 타입별 정확도 (검증된 defect bbox만)
    """
    total: int = Field(..., description="해당 타입 bbox 개수")
    correct: int = Field(..., description="정답 개수 (피드백 없음)")
    fp: int = Field(..., description="오탐 개수 (false_positive)")
    wrong: int = Field(..., description="클래스 오류 개수 (tp_wrong_class)")
    accuracy: float = Field(..., description="정확도 (%)")


class BboxStats(BaseModel):
    """
    bbox 단위 통계 (검증된 defect만)
    모델 정확도 분석의 핵심 지표
    """
    total: int = Field(..., description="전체 bbox 개수")
    correct: int = Field(..., description="정답 개수 (피드백 없음 = 암묵적 TP)")
    false_positive: int = Field(..., description="오탐 개수")
    wrong_class: int = Field(..., description="클래스 오류 개수")
    accuracy_rate: float = Field(..., description="정확도 (%)")
    by_defect_type: Dict[str, DefectTypeAccuracy] = Field(
        default_factory=dict,
        description="결함 타입별 정확도"
    )


class FeedbackTypeStats(BaseModel):
    """
    피드백 타입별 집계 (전체)
    """
    total: int = Field(..., description="전체 피드백 개수")
    false_positive: int = Field(default=0, description="오탐 개수")
    tp_wrong_class: int = Field(default=0, description="클래스 오류 개수")
    false_negative: int = Field(default=0, description="미탐 개수")


class ClassConfusion(BaseModel):
    """
    클래스 혼동 패턴 (FN 제외)
    모델이 어떤 클래스를 어떤 클래스로 잘못 예측하는지
    """
    from_class: str = Field(..., description="원본 예측 클래스")
    to_class: str = Field(..., description="실제 정답 클래스")
    count: int = Field(..., description="발생 횟수")


class FeedbackStatsResponse(BaseModel):
    """
    피드백 통계 응답 (bbox 기반 정확도 분석)

    MLOps 모니터링용:
    - image_stats: 전체 이미지 + 검증 진행률
    - bbox_stats: 검증된 defect의 bbox별 정확도 (핵심)
    - feedback_stats: 피드백 타입별 집계
    - class_confusion: 클래스 혼동 패턴
    """
    image_stats: ImageStats = Field(..., description="이미지 단위 통계")
    bbox_stats: BboxStats = Field(..., description="bbox 단위 통계 (검증된 defect만)")
    feedback_stats: FeedbackTypeStats = Field(..., description="피드백 타입별 집계")
    class_confusion: List[ClassConfusion] = Field(
        default_factory=list,
        description="클래스 혼동 패턴 (FN 제외)"
    )


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
    target_bbox: Optional[List[int]] = Field(None, description="대상 bbox [x1, y1, x2, y2]")


