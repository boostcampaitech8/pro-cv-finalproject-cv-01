# Pro-CV API Data Format Specification

본 문서는 `pro-cv` 프로젝트의 백엔드(`serving/api`)가 프론트엔드 및 엣지 디바이스와 주고받는 데이터의 형태를 정의합니다.

## 1. 개요

백엔드는 **FastAPI**로 구현되어 있으며, 데이터 검증을 위해 **Pydantic** 모델(Schemas)을 사용합니다.

**주요 통신 대상:**
- **Edge/Jetson**: 검사 결과를 전송하는 엣지 디바이스
- **Dashboard**: 실시간 모니터링 프론트엔드
- **MLOps**: 모델 성능 모니터링 및 피드백 시스템

**기술 스택:**
- Framework: FastAPI
- Database: SQLite (aiosqlite)
- Image Storage: AWS S3 (Presigned URL)
- Python: >=3.11

---

## 2. API 엔드포인트 목록

| 카테고리 | Method | Endpoint | 설명 |
|---------|--------|----------|------|
| Detect | POST | `/detect/` | 검사 결과 수신 |
| Stats | GET | `/stats` | 통계 정보 조회 |
| Stats | GET | `/latest` | 최근 검사 로그 |
| Stats | GET | `/defects` | 결함 타입별 집계 |
| Sessions | POST | `/sessions/` | 세션 생성 |
| Sessions | PATCH | `/sessions/{session_id}` | 세션 종료 |
| Sessions | GET | `/sessions/` | 세션 목록 조회 |
| Sessions | GET | `/sessions/{session_id}` | 특정 세션 조회 |
| Monitoring | GET | `/monitoring/health` | 시스템 건강 상태 |
| Monitoring | GET | `/monitoring/alerts` | 알림 조회 (경량) |
| Feedback | POST | `/feedback/` | 피드백 생성 |
| Feedback | GET | `/feedback/stats` | 피드백 통계 |
| Images | GET | `/raw/{file_path}` | 이미지 조회 (S3 리다이렉트) |

---

## 3. API 상세 데이터 포맷

### 3.1 검사 결과 수신 (Detect)

엣지 디바이스에서 검사 결과를 서버로 전송할 때 사용합니다. 한 장의 이미지에서 여러 개의 결함을 탐지할 수 있는 구조입니다.

- **Endpoint:** `POST /detect/`
- **Request Body (`DetectRequest`):**
    ```json
    {
      "timestamp": "string",       // ISO 8601 형식 (예: '2026-01-14T15:30:45')
      "image_id": "string",        // 이미지 식별자 (예: 'PCB_001234')
      "image": "string",           // (선택) Base64 인코딩된 이미지 데이터
      "detections": [              // 탐지된 결함 목록 (빈 배열 = 정상)
        {
          "defect_type": "string", // 결함 종류 (scratch, hole, contamination, crack)
          "confidence": 0.95,      // 신뢰도 (0.0 ~ 1.0)
          "bbox": [100, 50, 200, 150] // [x1, y1, x2, y2] 바운딩 박스
        }
      ],
      "session_id": 1              // (선택) 세션 ID
    }
    ```
- **Response Body (`DetectResponse`):**
    ```json
    {
      "status": "ok",
      "id": 123                    // 데이터베이스에 저장된 로그 ID
    }
    ```

**이미지 저장:**
- 불량 이미지(detections가 비어있지 않음)일 경우 AWS S3에 저장
- 저장 경로: `raw/{날짜}/{timestamp}_{image_id}.jpg`

#### Slack 알림 시스템 (내부 기능)

`POST /detect/` 호출 시 자동으로 트리거되는 **내부 알림 기능**입니다. 외부에서 호출하는 API가 아니라, 백엔드에서 Slack으로 **보내는** 웹훅입니다.

**동작 흐름:**
```
엣지 → POST /detect/ → DB 저장 → Health 상태 조회 → 상태 변화 감지 → Slack 웹훅 전송
```

**환경 변수 설정:**
| 변수 | 설명 | 필수 |
|------|------|------|
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL | O |
| `SLACK_ALERT_ENABLED` | 알림 활성화 (`true`/`false`) | O |

**상태 변화 감지:**

세션별로 이전 상태를 메모리에 저장하고, 상태가 변화할 때만 Slack 알림을 전송합니다.

| 이전 상태 | 현재 상태 | 메시지 |
|----------|----------|--------|
| (없음) | any | 🔔 세션 시작: 현재 상태 {status} |
| healthy | warning | ⚠️ 경고: 시스템 상태가 warning으로 전환 |
| healthy | critical | 🚨 긴급: 시스템 상태가 critical로 전환 |
| warning | critical | 🚨 악화: warning에서 critical로 악화 |
| warning | healthy | ✅ 개선: 정상 상태로 복구 |
| critical | warning | ⚠️ 부분 개선: critical에서 warning으로 |
| critical | healthy | ✅ 해결: 완전히 복구 |

**Slack 메시지 포맷:**
```
🚨 긴급: 시스템 상태가 critical로 전환되었습니다

• 상태: critical
• 세션: 세션 1 (활성)

🔴 Critical 알림:
• 불량률이 25.0%로 매우 높습니다
  → 생산 라인 점검 및 원인 분석 필요

🟡 Warning 알림:
• 평균 신뢰도가 0.83으로 낮습니다
  → 모델 재학습 검토 필요
```

**특징:**
- **이벤트 기반**: 스케줄러 없이 검사 결과 수신 즉시 동작
- **상태 변화 시에만 전송**: 알림 피로도 감소
- **세션별 상태 추적**: 세션 ID 기준으로 상태 관리
- **비동기 전송**: API 응답 지연 없이 백그라운드 처리

**관련 파일:**
- `routers/detect.py`: `check_and_send_slack_alert()` - 상태 변화 감지 로직
- `utils/slack_notifier.py`: `send_slack_alert()` - 웹훅 전송 로직
- `config/settings.py`: 환경 변수 로드

---

### 3.2 통계 정보 조회 (Stats)

프론트엔드 대시보드 구성을 위한 요약 통계 정보를 제공합니다.

- **Endpoint:** `GET /stats`
- **Query Parameter:**
  - `session_id` (선택): 특정 세션의 통계만 조회. 생략 시 전체 통계 반환.
- **Response Body (`StatsResponse`):**
    ```json
    {
      "total_inspections": 100,     // 총 검사 건수 (이미지 기준)
      "normal_count": 85,           // 정상 판정건수
      "defect_items": 15,           // 불량 판정건수 (이미지 기준)
      "total_defects": 22,          // 탐지된 총 결함 개수
      "defect_rate": 15.0,          // 불량률 (%)
      "avg_defects_per_item": 1.47, // 불량 이미지당 평균 결함 개수
      "avg_fps": 0.0,               // 평균 처리 FPS (현재 미구현)
      "last_defect": {              // 가장 최근 불량 정보 (또는 null)
        "id": 123,
        "timestamp": "2026-01-25T04:12:52",
        "image_id": "PCB_001",
        "result": "defect",
        "detections": [...],
        "image_path": "raw/20260125/...",
        "session_id": 1
      }
    }
    ```

---

### 3.3 최근 검사 로그 (Latest)

최근 N개의 검사 로그를 반환합니다.

- **Endpoint:** `GET /latest`
- **Query Parameters:**
  - `limit` (선택, 기본값: 10): 반환할 로그 개수
  - `session_id` (선택): 특정 세션의 로그만 조회
- **Response Body:** `List[Dict]`
    ```json
    [
      {
        "id": 123,
        "timestamp": "2026-01-25T04:12:52",
        "image_id": "PCB_001",
        "result": "defect",
        "detections": [
          {"defect_type": "scratch", "confidence": 0.95, "bbox": [10, 20, 100, 120]}
        ],
        "image_path": "raw/20260125/...",
        "session_id": 1
      }
    ]
    ```

---

### 3.4 결함 타입별 집계 (Defects)

결함 타입별 발생 횟수를 집계합니다.

- **Endpoint:** `GET /defects`
- **Query Parameter:**
  - `session_id` (선택): 특정 세션의 결함만 집계
- **Response Body:** `Dict[str, int]`
    ```json
    {
      "scratch": 15,
      "hole": 8,
      "contamination": 3,
      "crack": 2
    }
    ```

---

### 3.5 세션 관리 (Sessions)

검사 세션의 생성, 종료, 조회 기능을 제공합니다.

#### 3.5.1 세션 생성

- **Endpoint:** `POST /sessions/`
- **Response Body (`SessionCreateResponse`):** (201 Created)
    ```json
    {
      "id": 1,
      "started_at": "2026-01-25T10:00:00"
    }
    ```

#### 3.5.2 세션 종료

- **Endpoint:** `PATCH /sessions/{session_id}`
- **Response Body (`SessionResponse`):**
    ```json
    {
      "id": 1,
      "started_at": "2026-01-25T10:00:00",
      "ended_at": "2026-01-25T12:30:00"
    }
    ```
- **Error:** 404 Not Found (세션이 존재하지 않을 경우)

#### 3.5.3 세션 목록 조회

- **Endpoint:** `GET /sessions/`
- **Response Body (`SessionListResponse`):**
    ```json
    {
      "sessions": [
        {"id": 2, "started_at": "2026-01-26T09:00:00", "ended_at": null},
        {"id": 1, "started_at": "2026-01-25T10:00:00", "ended_at": "2026-01-25T12:30:00"}
      ]
    }
    ```

#### 3.5.4 특정 세션 조회

- **Endpoint:** `GET /sessions/{session_id}`
- **Response Body (`SessionResponse`):**
    ```json
    {
      "id": 1,
      "started_at": "2026-01-25T10:00:00",
      "ended_at": "2026-01-25T12:30:00"
    }
    ```
- **Error:** 404 Not Found

---

### 3.6 시스템 건강 상태 모니터링 (Health)

MLOps 모니터링을 위한 시스템 건강 상태 정보를 제공합니다.

- **Endpoint:** `GET /monitoring/health`
- **Query Parameter:**
  - `session_id` (선택): `"latest"`, 숫자, 또는 생략
- **Response Body (`HealthResponse`):**
    ```json
    {
      "status": "warning",           // healthy, warning, critical
      "timestamp": "2026-01-25T12:00:00",
      "session_info": {
        "id": 1,
        "started_at": "2026-01-25T10:00:00",
        "ended_at": null,
        "duration_seconds": 7200.0,
        "is_active": true
      },
      "total_inspections": 100,
      "normal_count": 85,
      "defect_count": 15,
      "defect_rate": 15.0,
      "total_defects": 22,
      "avg_defects_per_item": 1.47,
      "defect_confidence_stats": {
        "avg_confidence": 0.88,
        "min_confidence": 0.65,
        "max_confidence": 0.98,
        "distribution": {
          "high": 70,      // ≥0.9
          "medium": 20,    // 0.8~0.9
          "low": 8,        // 0.7~0.8
          "very_low": 2    // <0.7
        }
      },
      "defect_type_stats": [
        {"defect_type": "scratch", "count": 15, "avg_confidence": 0.88}
      ],
      "alerts": [
        {
          "level": "warning",
          "message": "불량률이 경고 수준입니다",
          "value": 15.0,
          "threshold": 10.0,
          "action": "생산 라인 점검 권장"
        }
      ]
    }
    ```

**알림 임계값:**

| 카테고리 | Warning | Critical |
|---------|---------|----------|
| 불량률 (%) | 10 | 20 |
| 평균 신뢰도 | 0.85 | 0.75 |
| 저신뢰도 비율 (%) | 20 | 40 |
| PCB당 평균 결함 | 2.0 | 3.0 |

---

### 3.7 알림 조회 (Alerts)

프론트엔드 폴링용 경량 알림 API입니다.

- **Endpoint:** `GET /monitoring/alerts`
- **Query Parameter:**
  - `session_id` (선택): `"latest"`, 숫자, 또는 생략
- **Response Body (`AlertsResponse`):**
    ```json
    {
      "status": "critical",
      "timestamp": "2026-01-25T12:00:00",
      "session_info": {
        "id": 1,
        "started_at": "2026-01-25T10:00:00",
        "ended_at": null,
        "duration_seconds": 7200.0,
        "is_active": true
      },
      "alerts": [
        {
          "level": "critical",
          "message": "불량률이 25.0%로 매우 높습니다",
          "value": 25.0,
          "threshold": 20.0,
          "action": "생산 라인 점검 및 원인 분석 필요"
        }
      ],
      "summary": {
        "defect_rate": 25.0,
        "avg_confidence": 0.88
      }
    }
    ```

---

### 3.8 피드백 시스템 (Feedback)

품질 관리자가 검사 결과를 검토하고 피드백을 제공하는 API입니다.

#### 3.8.1 피드백 생성

- **Endpoint:** `POST /feedback/`
- **Request Body (`FeedbackRequest`):**
    ```json
    {
      "log_id": 42,                          // 검사 로그 ID (필수)
      "feedback_type": "false_positive",     // 피드백 종류 (필수)
      "correct_label": "scratch",            // label_correction 시 필수
      "comment": "정상 PCB인데 먼지로 인해 오탐됨",
      "created_by": "manager_kim"
    }
    ```

**피드백 종류:**

| 타입 | 설명 | correct_label |
|------|------|---------------|
| `false_positive` | 정상인데 불량으로 오탐 | 불필요 |
| `false_negative` | 불량인데 정상으로 통과 | 불필요 |
| `label_correction` | 결함은 맞지만 종류가 틀림 | 필수 |

**correct_label 허용값:** `scratch`, `hole`, `contamination`, `crack`, `normal`

- **Response Body (`FeedbackResponse`):** (201 Created)
    ```json
    {
      "id": 1,
      "log_id": 42,
      "feedback_type": "false_positive",
      "correct_label": null,
      "comment": "정상 PCB인데 먼지로 인해 오탐됨",
      "created_at": "2026-01-30T10:30:45",
      "created_by": "manager_kim",
      "status": "ok"
    }
    ```

**에러 응답:**
- 404 Not Found: 존재하지 않는 log_id
- 422 Validation Error: correct_label 누락/잘못된 값

#### 3.8.2 피드백 통계 조회

- **Endpoint:** `GET /feedback/stats`
- **Response Body (`FeedbackStatsResponse`):**
    ```json
    {
      "total_feedback": 150,
      "by_type": {
        "false_positive": 80,
        "false_negative": 45,
        "label_correction": 25
      },
      "by_defect_type": [
        {
          "defect_type": "scratch",
          "false_positive": 50,
          "false_negative": 20,
          "label_correction": 10
        },
        {
          "defect_type": "hole",
          "false_positive": 30,
          "false_negative": 25,
          "label_correction": 15
        }
      ],
      "recent_feedback_count": 12,
      "period_description": "최근 24시간"
    }
    ```

---

### 3.9 이미지 조회 (Images)

S3에 저장된 결함 이미지를 Presigned URL로 조회합니다.

- **Endpoint:** `GET /raw/{file_path}`
- **Example:** `GET /raw/20260125/timestamp_PCB_001.jpg`
- **Response:** 307 Temporary Redirect → S3 Presigned URL

---

## 4. 데이터 스키마

### 4.1 검사 로그 테이블 (inspection_logs)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PRIMARY KEY |
| timestamp | TEXT | 검사 시각 (ISO 8601) |
| image_id | TEXT | 이미지 식별자 |
| result | TEXT | normal / defect |
| detections | TEXT | JSON 문자열 (결함 목록) |
| image_path | TEXT | S3 저장 경로 (nullable) |
| session_id | INTEGER | 세션 ID (FK, nullable) |

### 4.2 세션 테이블 (sessions)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PRIMARY KEY |
| started_at | TEXT | 세션 시작 시각 (ISO 8601) |
| ended_at | TEXT | 세션 종료 시각 (nullable) |

### 4.3 피드백 테이블 (feedback)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PRIMARY KEY |
| log_id | INTEGER | 검사 로그 ID (FK) |
| feedback_type | TEXT | 피드백 종류 |
| correct_label | TEXT | 올바른 라벨 (nullable) |
| comment | TEXT | 추가 설명 (nullable) |
| created_at | TEXT | 생성 시각 (ISO 8601) |
| created_by | TEXT | 작성자 (nullable) |

---

## 5. 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 | - |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 | - |
| `AWS_REGION` | AWS 리전 | `ap-northeast-2` |
| `S3_BUCKET_NAME` | S3 버킷 이름 | `pcb-data-storage` |
| `SLACK_WEBHOOK_URL` | Slack 웹훅 URL | - |
| `SLACK_ALERT_ENABLED` | Slack 알림 활성화 | `false` |

---

## 6. 결함 타입

| 코드 | 설명 |
|------|------|
| `scratch` | 스크래치 |
| `hole` | 구멍 |
| `contamination` | 오염 |
| `crack` | 크랙 |
