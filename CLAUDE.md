# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PCB 결함 탐지 Edge AI 시스템의 **백엔드 서버**.

- **역할**: 엣지(Jetson)에서 추론 결과를 받아 저장하고, 프론트(Streamlit)에 통계/이력 제공
- **기간**: 1개월 (마감 2월 11일)
- **팀**: 5명 중 백엔드 담당
- **배포**: AWS Lightsail (3.36.185.146)

```
[Jetson/엣지] ──POST /detect──→ [FastAPI/백엔드] ←──GET /stats── [Streamlit/프론트]
```

## Tech Stack

- **Framework**: FastAPI
- **Database**: SQLite (aiosqlite)
- **Image Storage**: 파일 시스템 (/images/defects/)
- **Python**: >=3.11
- **Deployment**: 1차 직접 설치 / 2차 Docker (준비됨)
- **Server**: AWS Lightsail (3.36.185.146)

## API Endpoints

### POST /detect/ (엣지 → 백엔드)

엣지에서 추론 결과 수신. **1개 PCB에서 여러 결함 지원**.

**Request Body:**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| timestamp | string | O | ISO 8601 형식 ("2026-01-14T15:30:45") |
| image_id | string | O | 이미지 식별자 ("PCB_001234") |
| image | string | X | Base64 인코딩 이미지 (불량일 때만) |
| detections | list | O | 결함 목록 (빈 배열 = 정상) |

**detections 배열 요소:**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| defect_type | string | O | 결함 종류 (scratch, hole 등) |
| confidence | float | O | 신뢰도 (0.0~1.0) |
| bbox | list[int] | O | [x1, y1, x2, y2] |

**예시 - 정상:**
```json
{
  "timestamp": "2026-01-18T15:00:00",
  "image_id": "PCB_001",
  "detections": []
}
```

**예시 - 불량 (결함 3개):**
```json
{
  "timestamp": "2026-01-18T15:01:00",
  "image_id": "PCB_002",
  "image": "base64_encoded_string",
  "detections": [
    {"defect_type": "scratch", "confidence": 0.95, "bbox": [10, 20, 100, 120]},
    {"defect_type": "scratch", "confidence": 0.87, "bbox": [150, 180, 200, 230]},
    {"defect_type": "hole", "confidence": 0.92, "bbox": [300, 350, 320, 380]}
  ]
}
```

**Response:**

```json
{"status": "ok", "id": 1}
```

### GET /stats (프론트 → 백엔드)

통계 정보 반환

**Response:**

| 필드 | 타입 | 설명 |
|------|------|------|
| total_inspections | int | 총 검사 수 (PCB 개수) |
| normal_count | int | 정상 PCB 개수 |
| defect_items | int | 불량 PCB 개수 |
| total_defects | int | 탐지된 결함 총 개수 |
| defect_rate | float | 불량률 (%) = defect_items / total_inspections × 100 |
| avg_defects_per_item | float | 불량 PCB당 평균 결함 개수 |
| avg_fps | float | 평균 FPS |
| last_defect | object | 가장 최근 불량 정보 |

**통계 개념:**
- `total_inspections`: 검사한 PCB 수 (DISTINCT image_id)
- `defect_items`: 1개 이상의 결함이 있는 PCB 수
- `total_defects`: 탐지된 결함의 총 개수 (1 PCB에 여러 결함 가능)
- `defect_rate`: 불량률 = (defect_items / total_inspections) × 100

**예시 응답:**
```json
{
  "total_inspections": 100,
  "normal_count": 90,
  "defect_items": 10,
  "total_defects": 25,
  "defect_rate": 10.0,
  "avg_defects_per_item": 2.5,
  "avg_fps": 0.0,
  "last_defect": {...}
}
```

**last_defect 객체:**

| 필드 | 타입 | 설명 |
|------|------|------|
| timestamp | string | 검사 시각 |
| image_id | string | 이미지 ID |
| result | string | "defect" |
| confidence | float | 신뢰도 |
| defect_type | string | 결함 종류 |
| bbox | list[int] | 결함 위치 |
| image_path | string | 이미지 경로 |

### GET /latest

최근 검사 이력 10개 반환

### GET /defects

결함 타입별 집계 반환

## Data Schema

**검사 로그 테이블 (inspection_logs):**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PRIMARY KEY |
| timestamp | TEXT | 검사 시각 |
| image_id | TEXT | 이미지 식별자 |
| result | TEXT | normal/defect |
| confidence | REAL | 신뢰도 |
| defect_type | TEXT | 결함 종류 (nullable) |
| bbox | TEXT | JSON 문자열 (nullable) |
| image_path | TEXT | 이미지 저장 경로 (nullable) |

## Development Setup

```bash
cd serving/api
uv sync --active
```

## Running the Server

**로컬 개발:**
```bash
cd serving/api
uv run uvicorn main:app --reload --port 8000
```

**서버 배포 (1차 - 직접 설치):**
```bash
cd serving/api
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

**서버 배포 (2차 - Docker, 추후):**
```bash
docker-compose up -d
```

API 문서 확인: http://localhost:8000/docs

## Project Structure

```
serving/api/
├── main.py              # FastAPI 앱, 라우터, lifespan
├── config/
│   └── settings.py      # 설정 중앙 관리
├── schemas/
│   └── schemas.py       # Pydantic 모델
├── routers/
│   ├── detect.py        # POST /detect
│   └── stats.py         # GET /stats, /latest, /defects
├── database/
│   └── db.py            # SQLite (aiosqlite)
├── utils/
│   └── image_utils.py   # Base64 디코딩, 이미지 저장
├── data/
│   └── inspection.db    # SQLite 파일
├── images/
│   └── defects/         # 결함 이미지
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pyproject.toml
└── uv.lock
```

## Implementation Priority

1. ✅ **완료**: POST /detect + SQLite 저장
2. ✅ **완료**: GET /stats, /latest, /defects
3. ✅ **완료**: 이미지 파일 저장 + Base64 처리
4. ✅ **완료**: 에러 핸들링, Config 모듈
5. ✅ **준비됨**: Docker 설정 (2차 배포용)
6. ✅ **완료**: 1차 서버 배포 (직접 설치, 2026-01-18)
7. ⏳ **예정**: 테스트 코드, 로깅
8. ⏳ **선택**: 2차 Docker 전환

## Important Notes

- **CORS 설정 필수**: 프론트엔드 도메인에서 API 호출 허용
- **이미지 저장**: DB에 직접 저장 X → 파일 시스템에 저장, DB엔 경로만
- **Base64 디코딩**: 불량 이미지 수신 시 디코딩해서 파일로 저장
- **Static Files**: 저장된 이미지를 프론트에서 불러올 수 있게 서빙
- **데이터 영속화**: 서버 로컬 디렉토리 (1차) / Docker 볼륨 마운트 (2차)
- **포트 노출**: Lightsail 방화벽에서 8000 포트 개방 필요

## Collaboration

**엣지 담당과 협의:**
- 전송 주기 : 매 프레임
- 비동기 전송
- 전송 실패 시 처리 방식 : 데이터 유실

**프론트 담당과 협의:**
- 폴링 주기 : 현재 1초
- 추가로 필요한 통계 데이터 : 향후 정함
- 시간별/일별 그래프용 데이터 필요 여부 : 향후 정함

**코드 주석은 한국어로 작성해줘**

## Deployment

### 서버 정보
- **Host**: pcb-defect
- **IP**: 3.36.185.146
- **User**: ubuntu
- **접속 키**: ~/.ssh/LightsailDefaultKey-ap-northeast-2.pem

### 서버 접속
```bash
ssh pcb-defect
# 또는
ssh -i ~/.ssh/LightsailDefaultKey-ap-northeast-2.pem ubuntu@3.36.185.146
```

### 1차 배포: 직접 설치 (✅ 완료)

**배포 일시**: 2026-01-18
**브랜치**: feat/BE
**실행 방식**: nohup (백그라운드)

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

**코드 업데이트 후 재시작:**
```bash
ssh pcb-defect
cd ~/pro-cv-finalproject-cv-01
git pull origin feat/BE
cd serving/api
uv sync
pkill -f uvicorn
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

### 2차 배포: Docker (추후)
안정화 후 Docker로 전환 예정

**준비된 파일:**
- Dockerfile (로컬에만 보관, Git에 미포함)
- docker-compose.yml (로컬에만 보관, Git에 미포함)
- .dockerignore (로컬에만 보관, Git에 미포함)

### 배포 URL
- **API 문서**: http://3.36.185.146:8000/docs
- **헬스체크**: http://3.36.185.146:8000/stats
- **Jetson → POST**: http://3.36.185.146:8000/detect
- **Streamlit → GET**: http://3.36.185.146:8000/stats

### 팀 연동 안내

**Jetson 엣지 팀:**
- 추론 결과를 `http://3.36.185.146:8000/detect`로 POST 전송

**Streamlit 대시보드 팀:**
- 통계 데이터를 `http://3.36.185.146:8000/stats`에서 GET 조회
- 최근 로그: `http://3.36.185.146:8000/latest?limit=10`
- 결함 목록: `http://3.36.185.146:8000/defects`

## Commit Convention

커밋 메시지 작성 시 `.gitmessage` 템플릿을 따를 것.

**형식:**
```
[파트] 타입: 제목 (50자 이내)

왜 변경했나요? (선택사항)

특이사항/영향받는 부분 (선택사항)
```

**파트:**
- `BE` : 백엔드 (FastAPI)
- `Edge` : 엣지/추론 (Jetson)
- `FE` : 프론트엔드 (Streamlit)
- `Train` : 모델 학습
- `Docs` : 문서
- `Config` : 설정/환경

**타입:**
- `feat` : 새 기능 추가
- `fix` : 버그 수정
- `refactor` : 코드 리팩토링
- `test` : 테스트 추가/수정
- `docs` : 문서 수정
- `style` : 코드 포맷팅
- `chore` : 빌드, 설정 파일 수정

**예시:**
```
[BE] feat: POST /detect 엔드포인트 구현
[Edge] fix: Base64 인코딩 오류 수정
[FE] feat: 실시간 그래프 추가
```