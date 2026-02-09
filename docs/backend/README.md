# 백엔드 개요

PCB 결함 탐지 시스템의 FastAPI 기반 백엔드 서버입니다.

## 기술 스택

| 구분 | 기술 | 선정 이유 |
|------|------|-----------|
| 웹 프레임워크 | FastAPI | 비동기 I/O 지원, 자동 API 문서 생성 |
| 데이터베이스 | SQLite (aiosqlite) | 임베디드 DB, 비동기 지원 |
| 이미지 저장 | AWS S3 | 대용량 이미지 저장, 학습 서버 연동 |
| 실시간 알림 | Slack Webhook | 팀 협업 도구 통합 |
| 데이터 검증 | Pydantic v2 | 타입 안전성, 교차 필드 검증 |
| 패키지 관리 | uv | 빠른 의존성 해결 |

## 프로젝트 구조

```
serving/api/
├── main.py                    # FastAPI 앱, 라우터 등록
├── config/
│   └── settings.py            # 환경변수, 알림 임계값
├── routers/
│   ├── detect.py              # POST /detect
│   ├── stats.py               # GET /stats, /latest, /defects
│   ├── sessions.py            # 세션 관리
│   ├── monitoring.py          # GET /monitoring/health, /alerts
│   ├── feedback.py            # 피드백 시스템
│   └── images.py              # GET /raw/{path}
├── schemas/
│   └── schemas.py             # Pydantic 모델
├── database/
│   └── db.py                  # SQLite CRUD
├── utils/
│   ├── auth.py                # API Key 인증
│   ├── image_utils.py         # S3 업로드, Presigned URL
│   ├── slack_notifier.py      # Slack 웹훅
│   └── s3_dataset.py          # S3 데이터셋 관리
├── data/
│   └── inspection.db          # SQLite 파일
└── pyproject.toml
```

## 데이터베이스 설계

### inspection_logs 테이블

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER PK | 자동 증가 |
| timestamp | TEXT | 검사 시각 (ISO 8601) |
| image_id | TEXT | 이미지 식별자 |
| result | TEXT | "normal" 또는 "defect" |
| detections | TEXT | 결함 목록 JSON 배열 |
| image_path | TEXT | S3 이미지 키 |
| session_id | INTEGER FK | 세션 외래키 |
| is_verified | BOOLEAN | 검증 완료 여부 |
| verified_at | TEXT | 검증 시각 |
| verified_by | TEXT | 검증자 |

### sessions 테이블

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER PK | 자동 증가 |
| started_at | TEXT | 세션 시작 시각 |
| ended_at | TEXT | 세션 종료 시각 |

### feedback 테이블

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER PK | 자동 증가 |
| log_id | INTEGER FK | inspection_logs 외래키 |
| feedback_type | TEXT | false_positive, tp_wrong_class, false_negative |
| correct_label | TEXT | 올바른 라벨 |
| comment | TEXT | 추가 설명 |
| created_at | TEXT | 생성 시각 |
| created_by | TEXT | 작성자 |
| status | TEXT | pending / resolved |
| target_bbox | TEXT | 대상 bbox JSON |

## 환경 변수

| 변수 | 용도 | 필수 |
|------|------|------|
| EDGE_API_KEY | 엣지 디바이스 인증 | O |
| AWS_ACCESS_KEY_ID | S3 접근 | O |
| AWS_SECRET_ACCESS_KEY | S3 접근 | O |
| SLACK_WEBHOOK_URL | Slack 알림 | X |
| SLACK_ALERT_ENABLED | 알림 활성화 | X |

## 하위 문서

| 문서 | 설명 |
|------|------|
| [api.md](api.md) | API 엔드포인트 명세 |
| [monitoring.md](monitoring.md) | MLOps 모니터링 시스템 |
| [feedback.md](feedback.md) | 피드백 및 자동 재라벨링 |

## 실행 방법

```bash
cd serving/api
uv sync --active
uv run uvicorn main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

## 배포 정보

| 구분 | 내용 |
|------|------|
| 서버 | EC2 (3.35.182.98) |
| 포트 | 8080 |
| 실행 방식 | systemd 서비스 |
| 서비스명 | fastapi.service |
