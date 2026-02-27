"""
세션 관리 API 라우터

세션 생성, 종료, 목록 조회 기능 제공
"""
from fastapi import APIRouter, HTTPException, status
from schemas.schemas import SessionResponse, SessionCreateRequest, SessionCreateResponse, SessionListResponse
from database import db

router = APIRouter(
    prefix="/sessions",
    tags=["Sessions"]
)


@router.post("/", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: SessionCreateRequest):
    """
    새 세션을 시작합니다.
    엣지에서 추론 시작 시 호출.
    모델 교체 시 새 세션을 생성하여 모델별 추적 가능.
    """
    try:
        session = await db.create_session(model_name=request.model_name)
        return SessionCreateResponse(
            id=session["id"],
            started_at=session["started_at"],
            model_name=session["model_name"]
        )
    except Exception as e:
        print(f"[Sessions] create_session 에러: {e}")
        raise HTTPException(status_code=500, detail=f"세션 생성 실패: {str(e)}")


@router.patch("/{session_id}", response_model=SessionResponse)
async def end_session(session_id: int):
    """
    세션을 종료합니다 (ended_at 설정).
    엣지에서 추론 종료 시 호출.
    """
    session = await db.end_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return SessionResponse(**session)


@router.get("/", response_model=SessionListResponse)
async def get_sessions():
    """
    모든 세션 목록을 반환합니다 (최신순).
    """
    sessions = await db.get_sessions()
    return SessionListResponse(
        sessions=[SessionResponse(**s) for s in sessions]
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: int):
    """
    특정 세션 정보를 반환합니다.
    """
    session = await db.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return SessionResponse(**session)
