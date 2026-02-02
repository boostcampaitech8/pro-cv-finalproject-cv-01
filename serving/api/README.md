# PCB Defect Detection Backend API

이 프로젝트는 엣지(Jetson) 디바이스로부터 결함 탐지 결과를 실시간으로 수신하고, 이를 데이터베이스에 저장하며 대시보드에 데이터를 제공하는 Backend API 서버입니다.

## 📂 폴더 및 파일 구조

### 1. Root Files
- `main.py`: FastAPI 서버의 진입점. 서버 설정, 미들웨어 등록, 라우터 연결 및 DB 초기화를 담당합니다.
- `pyproject.toml` & `requirements.txt`: Python 의존성 관리 및 프로젝트 설정 파일입니다.
- `uv.lock`: `uv` 패키지 매니저를 사용할 경우 생성되는 의존성 잠금 파일입니다.

### 2. 주요 디렉토리
- `routers/`: API 엔드포인트별 로직이 분리되어 있습니다.
  - `detect.py`: 엣지 디바이스로부터 탐지 결과를 수신 (`/detect`)
  - `stats.py`: 통계 데이터 및 추이, 유형벌 집계 데이터 제공 (`/stats`, `/latest`, `/defects`)
  - `sessions.py`: 추론 세션 관리 (시작, 종료, 목록 조회) (`/sessions`)
- `database/`: DB 연결 및 쿼리 로직이 포함되어 있습니다.
  - `db.py`: SQLite(aiosqlite)를 사용한 비동기 DB 작업 처리 (검사 로그 저장, 세션 관리 등)
- `schemas/`: Pydantic을 이용한 요청(Request) 및 응답(Response) 데이터 모델링이 정의되어 있습니다.
- `config/`: 프로젝트 전역 설정이 포함되어 있습니다.
  - `settings.py`: DB 경로, 이미지 저장 경로, CORS 설정 등
- `utils/`: 공통 유틸리티 함수입니다.
  - `image_utils.py`: 이미지 Base64 디코딩 및 디스크 저장 로직
- `images/`: 결함으로 판정된 원본 이미지들이 저장되는 경로입니다.

## 🔗 시스템 연결 흐름

### 1. Edge (Jetson) -> Backend
- 엣지에서 PCB 분석이 완료되면 `/detect` 엔드포인트로 결과를 전송합니다.
- **결함 발견 시**: 이미지를 Base64로 포함하여 전송하고, 백엔드는 이를 `images/` 폴더에 저장합니다.
- **DB 저장**: 모든 검사 결과는 `inspection_logs` 테이블에 기록됩니다.

### 2. Backend -> Dashboard (Frontend)
- 대시보드는 1초마다 `/stats`, `/defects`, `/latest` 등의 API를 호출하여 최신 데이터를 가져갑니다.
- 백엔드는 DB 내의 데이터를 집계하여 대시보드가 차트를 그릴 수 있는 형태로 반환합니다.

## 🚀 주요 API 명세 (Endpoint)

| Method | Endpoint | Description | 담당 파일 |
| :--- | :--- | :--- | :--- |
| `POST` | `/detect/` | 엣지 결과 수신 및 로그/이미지 저장 | `routers/detect.py` |
| `GET` | `/stats` | 전체 및 세션별 검사 통계 조회 | `routers/stats.py` |
| `GET` | `/defects` | 결함 유형별 집계 데이터 조회 | `routers/stats.py` |
| `POST` | `/sessions/` | 새로운 추론 세션 시작 | `routers/sessions.py` |
| `GET` | `/sessions/` | 전체 세션 목록 조회 | `routers/sessions.py` |

## 🛠 실행 방법

1.  **의존성 설치**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **서버 실행**:
    ```bash
    # uvicorn 사용 시
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
3.  **API 문서 확인**:
    - 서버 실행 후 `http://localhost:8000/docs` 접속 시 Swagger UI를 통해 모든 API를 테스트해 볼 수 있습니다.
