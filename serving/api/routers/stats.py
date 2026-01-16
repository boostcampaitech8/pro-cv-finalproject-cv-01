from fastapi import APIRouter
from schemas.schemas import StatsResponse
from database import db
from typing import List, Dict

router = APIRouter(
    tags=["Stats"]
)

@router.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """
    Returns current statistics: total count, defect count, rate, etc.
    Required by Streamlit frontend.
    """
    return await db.get_stats()

@router.get("/latest", response_model=List[Dict])
async def get_latest_logs(limit: int = 10):
    """
    Returns the N most recent inspection logs.
    """
    return await db.get_recent_logs(limit)

@router.get("/defects", response_model=Dict[str, int])
async def get_defect_aggregation():
    """
    Returns aggregation of defect types.
    Example: {"scratch": 5, "dent": 2}
    """
    # This logic could be moved to db.py if complex
    defect_logs = await db.get_defect_logs()
    
    aggregation = {}
    for log in defect_logs:
        d_type = log.get("defect_type")
        if d_type:
            aggregation[d_type] = aggregation.get(d_type, 0) + 1
        else:
            # Handle cases where defect_type might be missing for a defect
            aggregation["unknown"] = aggregation.get("unknown", 0) + 1
            
    return aggregation
