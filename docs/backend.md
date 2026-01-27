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
| Docker 설정 | ✅ 준비됨 (2차 배포용) |
| 1차 서버 배포 (직접 설치) | ✅ 완료 (2026-01-18) |
| 다중 결함 지원 API 변경 | ✅ 완료 (2026-01-18) |
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

## 배포 환경

### AWS Lightsail 서버
- **Host**: pcb-defect
- **IP**: 3.36.185.146
- **User**: ubuntu
- **Spec**: $12 Plan (2GB RAM, 2 vCPUs, 60GB SSD)

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
| `DetectRequest` | timestamp, image_id, image (선택), detections (결함 배열) |
| `Detection` | defect_type, confidence, bbox (개별 결함 정보) |
| `DetectResponse` | status, id |
| `StatsResponse` | total_inspections, normal_count, defect_items, total_defects, defect_rate, avg_defects_per_item, avg_fps, last_defect |
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

## Docker 구성 (2차 배포용 - 준비됨)

### 1. Dockerfile (`serving/api/Dockerfile`)
- Base Image: `python:3.11-slim`
- 패키지 관리자: uv
- 포트: 8000
- 볼륨 마운트: data/, images/

### 2. docker-compose.yml
- 서비스: backend
- 포트 매핑: 8000:8000
- 볼륨 영속화: SQLite DB, 결함 이미지

### 3. .dockerignore
- .venv, __pycache__, *.pyc 제외

## 서버 실행 방법

```bash
cd serving/api
uv sync
uv run uvicorn main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

## 서버 배포 가이드 (1차 - 직접 설치)

자세한 내용은 [deploy.md](deploy.md) 참고

### 1. SSH 접속
```bash
ssh -i ~/.ssh/LightsailDefaultKey-ap-northeast-2.pem ubuntu@3.36.185.146
```

### 2. Python/uv 설치
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 python3.11-venv python3-pip -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3. 리포지토리 클론
```bash
git clone https://github.com/boostcampaitech8/pro-cv-finalproject-cv-01.git
cd pro-cv-finalproject-cv-01
git checkout dev
```

### 4. 서버 실행
```bash
cd serving/api
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. 방화벽 설정
Lightsail 콘솔 → 네트워킹 탭:
- TCP 8000 (FastAPI)
- TCP 8501 (Streamlit - 추후)

### 6. 배포 확인
- API 문서: http://3.36.185.146:8000/docs
- 헬스체크: http://3.36.185.146:8000/stats

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

## 배포 현황 (2026-01-18)

### ✅ 1차 배포 완료

- **배포 일시**: 2026-01-18
- **배포 방식**: 직접 설치 (uv + Python 3.11)
- **서버 주소**: http://3.36.185.146:8000
- **브랜치**: feat/BE
- **실행 방식**: nohup (백그라운드 실행)
- **API 테스트**: GET /stats 정상 작동 확인

### 팀 연동 준비

**Jetson 엣지 팀:**
- 엔드포인트: `http://3.36.185.146:8000/detect`
- 메서드: POST
- 추론 결과를 이 주소로 전송하면 됨

**Streamlit 대시보드 팀:**
- 통계 API: `http://3.36.185.146:8000/stats`
- 최근 로그: `http://3.36.185.146:8000/latest?limit=10`
- 결함 목록: `http://3.36.185.146:8000/defects`

### 서버 관리 명령어

**서버 시작:**
```bash
ssh pcb-defect
cd ~/pro-cv-finalproject-cv-01/serving/api
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

**서버 중지:**
```bash
ssh pcb-defect
pkill -f uvicorn
```

**서버 상태 확인:**
```bash
ssh pcb-defect
ps aux | grep uvicorn
```

**코드 업데이트:**
```bash
ssh pcb-defect
cd ~/pro-cv-finalproject-cv-01
git pull origin feat/BE
cd serving/api
uv sync
pkill -f uvicorn
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

## 다음 단계

1. ~~**1차 배포**: AWS Lightsail에 직접 설치 방식으로 배포~~ ✅ 완료
2. ~~**다중 결함 지원**: 1개 PCB에서 여러 결함 탐지 지원~~ ✅ 완료 (2026-01-18)
3. **팀 연동 테스트**: Jetson, Streamlit과 연동 확인
4. **테스트 코드**: `tests/` 폴더에 API 테스트 스크립트 추가
5. **로깅**: API 요청/응답 로깅 구현
6. **avg_fps 계산**: 엣지 팀과 협의 후 실제 FPS 계산 로직 추가
7. **2차 배포 (선택)**: Docker로 전환 (안정화 후)
8. **모니터링**: 로그 수집, 헬스체크 설정

## API 변경 이력

### 2026-01-18: 다중 결함 지원

**변경 사항:**
- `POST /detect/`: detections 배열로 여러 결함 한 번에 전송 가능
- `GET /stats`: 통계 지표 개선 (total_inspections, defect_items, total_defects 분리)

**새로운 Request 구조:**
```json
{
  "timestamp": "2026-01-18T15:00:00",
  "image_id": "PCB_001",
  "detections": [
    {"defect_type": "scratch", "confidence": 0.95, "bbox": [10, 20, 100, 120]},
    {"defect_type": "hole", "confidence": 0.87, "bbox": [150, 180, 200, 230]}
  ]
}
```

**새로운 통계 개념:**
- `total_inspections`: 검사한 PCB 개수 (DISTINCT image_id)
- `defect_items`: 불량 PCB 개수
- `total_defects`: 탐지된 결함 총 개수
- `defect_rate`: 불량률 = (defect_items / total_inspections) × 100
