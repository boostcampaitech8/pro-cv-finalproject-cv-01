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


# --- Request Models ---

class DetectRequest(BaseModel):
    """
    Request body for POST /detect
    """
    timestamp: str = Field(..., description="ISO 8601 timestamp, e.g., '2026-01-14T15:30:45'")
    image_id: str = Field(..., description="Image Identifier, e.g., 'PCB_001234'")
    result: str = Field(..., description="'normal' or 'defect'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    
    # Optional fields (required/relevant only if result == 'defect')
    defect_type: Optional[str] = None
    bbox: Optional[List[int]] = None
    image: Optional[str] = Field(None, description="Base64 encoded image string")


# --- Response Models ---

class DetectResponse(BaseModel):
    """
    Response for POST /detect
    """
    status: str = "ok"
    id: int


class StatsResponse(BaseModel):
    """
    Response for GET /stats
    """
    total_count: int
    normal_count: int
    defect_count: int
    defect_rate: float = Field(..., description="Percentage of defects (0-100)")
    avg_fps: float = Field(..., description="Average processing/transmission FPS")
    last_defect: Optional[DefectInfo] = Field(None, description="Details of the most recent defect")
