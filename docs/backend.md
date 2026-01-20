# 구현 계획 - PCB 결함 탐지 백엔드

이 계획서는 PCB 결함 탐지 시스템을 위한 모듈화된 FastAPI 백엔드 구축 단계를 설명합니다.

## 진행 상황

| 항목 | 상태 |
|------|------|
| 프로젝트 구조 설계 | ✅ 완료 |
| 데이터 모델 (schemas.py) | ✅ 완료 |
| 데이터베이스 계층 (db.py) | ✅ 완료 (SQLite) |
| API 라우터 (detect, stats) | ✅ 완료 |
| 메인 애플리케이션 | ✅ 완료 |
| 에러 핸들링 | ✅ 완료 |
| API 테스트 | ✅ 완료 |
| SQLite 전환 | ✅ 완료 |
| Config 모듈 | ✅ 완료 |
| 테스트 코드 | ⏳ 예정 |
| 로깅 | ⏳ 예정 |

## 프로젝트 구조 (모노레포)

```
project/
├── .github/
├── .gitignore
├── .gitmessage              # 커밋 메시지 템플릿
├── CLAUDE.md
├── README.md
├── docs/
│   ├── README.md               # 문서 허브
│   ├── backend.md              # 백엔드 계획
│   ├── frontend.md             # 프론트엔드 계획
│   ├── edge.md                 # 엣지 추론 계획
│   └── training.md             # 모델 학습 계획
├── serving/
│   └── api/                    # 백엔드 (FastAPI)
│       ├── .venv/              # 가상환경
│       ├── main.py             # 앱 설정, CORS, 라우터 통합
│       ├── config/
│       │   └── settings.py     # 설정값 중앙 관리
│       ├── schemas/
│       │   └── schemas.py      # Pydantic 요청/응답 모델
│       ├── routers/
│       │   ├── detect.py       # POST /detect
│       │   └── stats.py        # GET /stats, /latest, /defects
│       ├── database/
│       │   └── db.py           # SQLite 연동 (aiosqlite)
│       ├── data/
│       │   └── inspection.db   # SQLite 데이터베이스 파일
│       ├── utils/
│       │   └── image_utils.py  # Base64 디코딩, 이미지 저장
│       ├── images/
│       │   └── defects/        # 결함 이미지 저장 경로
│       ├── pyproject.toml
│       └── uv.lock
│   └── edge/                   # 엣지 추론 (예정)
├── training/                   # 모델 학습 (예정)
└── dashboard/                  # 프론트엔드 (예정)
```

## 구현 완료 내역

### 1. Config 모듈 (`serving/api/config/settings.py`)

| 설정 | 설명 |
|------|------|
| `DB_PATH` | SQLite 데이터베이스 경로 |
| `IMAGE_DIR` | 결함 이미지 저장 경로 |
| `CORS_ORIGINS` | CORS 허용 도메인 |
| `API_TITLE`, `API_VERSION` | API 메타정보 |

### 2. 데이터 모델 (`serving/api/schemas/schemas.py`)

| 모델 | 설명 |
|------|------|
| `DetectRequest` | timestamp, image_id, result, confidence + 선택적 필드 (defect_type, bbox, image) |
| `DetectResponse` | status, id |
| `StatsResponse` | total_count, normal_count, defect_count, defect_rate, avg_fps, last_defect |
| `DefectInfo` | 결함 상세 정보 객체 |

### 3. 데이터베이스 계층 (`serving/api/database/db.py`)

- **저장소**: SQLite (aiosqlite)
- **테이블**: `inspection_logs`
- **주요 함수**:
  - `init_db()` - 테이블 생성 (서버 시작 시)
  - `add_inspection_log()` - 검사 결과 저장
  - `get_stats()` - 통계 계산
  - `get_recent_logs()` - 최근 로그 조회
  - `get_defect_logs()` - 결함 로그만 조회

### 4. API 라우터

**`serving/api/routers/detect.py`**
- `POST /detect/`: 결함 탐지 결과 수신, 이미지 저장, 로그 기록

**`serving/api/routers/stats.py`**
- `GET /stats`: 통계 정보 반환
- `GET /latest`: 최근 검사 이력 반환
- `GET /defects`: 결함 타입별 집계 반환

### 5. 유틸리티 (`serving/api/utils/image_utils.py`)

- `decode_base64_image(base64_string: str) -> bytes`
- `save_defect_image(image_data: bytes, image_id: str, timestamp: str) -> str`
- 파일명 규칙: `{timestamp}_{image_id}.jpg` (timestamp의 `:` → `-` 변환)

### 6. 에러 핸들링

| 상황 | HTTP 상태 코드 |
|------|---------------|
| 필수 필드 누락 | 422 Validation Error |
| Base64 디코딩 실패 | 400 Bad Request |
| 파일 저장 실패 | 500 Internal Server Error |

## 서버 실행 방법

```bash
cd serving/api
uv sync
uv run uvicorn main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

## Git 브랜치 전략

```
main        ← 안정 버전
└── dev     ← 개발 통합
     └── feat/JE  ← 백엔드 작업 (현재)
```

## 커밋 컨벤션

`.gitmessage` 템플릿 사용:
```
[파트] 타입: 제목

예시:
[BE] feat: POST /detect 엔드포인트 구현
[BE] fix: Base64 디코딩 오류 수정
```

## 다음 단계

1. **테스트 코드**: `tests/` 폴더에 API 테스트 스크립트 추가
2. **로깅**: API 요청/응답 로깅 구현
3. **avg_fps 계산**: 엣지 팀과 협의 후 실제 FPS 계산 로직 추가
