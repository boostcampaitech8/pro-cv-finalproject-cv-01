from pydantic import BaseModel, Field
from typing import Optional, List

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
