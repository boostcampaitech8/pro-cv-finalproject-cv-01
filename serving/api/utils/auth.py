"""
API Key 인증 모듈

Edge Device가 백엔드 API에 접근할 때 사용하는 API Key를 검증합니다.
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config import settings

# HTTP 헤더에서 X-API-KEY를 추출하는 스키마
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    HTTP 요청 헤더의 X-API-KEY를 검증합니다.
    
    Args:
        api_key: HTTP 헤더에서 추출된 API Key
        
    Returns:
        str: 검증된 API Key
        
    Raises:
        HTTPException: API Key가 없거나 잘못된 경우 403 Forbidden
    """
    # API Key가 설정되지 않은 경우 (개발 환경)
    if settings.EDGE_API_KEY is None:
        # 경고 메시지 출력 (프로덕션에서는 이 경우 서버가 시작되지 않아야 함)
        return None
    
    # API Key가 제공되지 않은 경우
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key가 필요합니다. X-API-KEY 헤더를 포함해주세요."
        )
    
    # API Key가 일치하지 않는 경우
    if api_key != settings.EDGE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="유효하지 않은 API Key입니다."
        )
    
    return api_key
