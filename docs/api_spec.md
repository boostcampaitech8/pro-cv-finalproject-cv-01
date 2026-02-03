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
| Feedback | POST | `/feedback/bulk` | 다중 bbox 피드백 + 자동 재라벨링 |
| Feedback | GET | `/feedback/stats` | 피드백 통계 (bbox 기반 정확도) |
| Feedback | GET | `/feedback/queue` | 라벨링 대기열 조회 |
| Feedback | GET | `/feedback/export` | 데이터셋 내보내기 정보 |
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
      "image": "string",           // (선택) Base64 인코딩된 이미지 데이터 (정상/불량 관계없이 전송 가능)
      "detections": [              // 탐지된 결함 목록 (빈 배열일 경우 '정상')
        {
          "defect_type": "string", // 결함 종류 (scratch, hole, contamination, crack)
          "confidence": 0.95,      // 신뢰도 (0.0 ~ 1.0)
          "bbox": [100, 50, 200, 150] // [x1, y1, x2, y2] 바운딩 박스
        }
      ],
      "session_id": 1              // (선택) 세션 ID - 검사 결과를 특정 세션에 연결
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
- **Query Parameter:** `session_id` (선택) - 특정 세션의 통계만 조회. 생략 시 전체 통계 반환.
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
        "detections": [
          {
            "defect_type": "defect_5",
            "confidence": 0.8187,
            "bbox": [627, 542, 724, 612]
          },
          {
            "defect_type": "defect_5",
            "confidence": 0.7866,
            "bbox": [859, 1114, 911, 1173]
          }
        ],
        "image_path": "images/defects/2026-01-25T04-12-52.046352_09_spur_08.jpg.jpg",
        "session_id": 1              // 세션 ID (null일 수 있음)
      }
    }
    ```

---

### 3.3 최근 검사 로그 (Latest)

최근 N개의 검사 로그를 반환합니다.

- **Endpoint:** `GET /latest`
- **Query Parameters:**
    - `limit` (기본값: 10) - 반환할 로그 개수
    - `session_id` (선택) - 특정 세션의 로그만 조회. 생략 시 전체 로그 반환.
- **Response Body (List of InspectionLogResponse):**
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
        "image_path": "images/defects/2026-01-25T04-12-52.046352_09_spur_08.jpg.jpg",
        "session_id": 1              // 세션 ID (null일 수 있음)
      },
      ...
    ]
    ```

---

### 3.4 결함 타입별 집계 (Defects)

결함 타입별 발생 횟수를 집계합니다.

- **Endpoint:** `GET /defects`
- **Query Parameter:** `session_id` (선택) - 특정 세션의 결함만 집계. 생략 시 전체 결함 집계.
- **Response Body:**
    ```json
    {
      "scratch": 15,
      "hole": 8,
      "contamination": 3,
      "crack": 2
    }
    ```

---

### 2.5 세션 관리 (Sessions)
검사 세션을 생성하고 관리합니다. 세션은 엣지에서 추론을 시작/종료할 때 생성/종료되며, 같은 세션 내의 검사 결과를 그룹으로 관리할 수 있습니다.

#### POST /sessions/ - 세션 생성
새 세션을 시작합니다. 엣지에서 추론 시작 시 호출합니다.

- **Endpoint:** `POST /sessions/`
- **Request Body:** 없음
- **Response Body (`SessionCreateResponse`):**
    ```json
    {
      "id": 1,                           // 생성된 세션 ID
      "started_at": "2026-01-30T10:00:00.123456"  // 세션 시작 시간 (ISO 8601)
    }
    ```
- **Status Code:** `201 Created`

---

#### PATCH /sessions/{session_id} - 세션 종료
세션을 종료합니다 (ended_at 설정). 엣지에서 추론 종료 시 호출합니다.

- **Endpoint:** `PATCH /sessions/{session_id}`
- **Path Parameter:** `session_id` (정수) - 종료할 세션 ID
- **Request Body:** 없음
- **Response Body (`SessionResponse`):**
    ```json
    {
      "id": 1,
      "started_at": "2026-01-30T10:00:00.123456",
      "ended_at": "2026-01-30T11:30:00.654321"   // 세션 종료 시간 (ISO 8601)
    }
    ```
- **Error Response:**
    - `404 Not Found`: 세션이 존재하지 않는 경우
    ```json
    {"detail": "Session 999 not found"}
    ```

---

#### GET /sessions/ - 세션 목록 조회
모든 세션 목록을 반환합니다 (최신순 정렬).

- **Endpoint:** `GET /sessions/`
- **Response Body (`SessionListResponse`):**
    ```json
    {
      "sessions": [
        {
          "id": 3,
          "started_at": "2026-01-30T14:00:00.000000",
          "ended_at": null                         // 진행 중인 세션
        },
        {
          "id": 2,
          "started_at": "2026-01-30T12:00:00.000000",
          "ended_at": "2026-01-30T13:30:00.000000"
        },
        {
          "id": 1,
          "started_at": "2026-01-30T10:00:00.000000",
          "ended_at": "2026-01-30T11:30:00.000000"
        }
      ]
    }
    ```

---

#### GET /sessions/{session_id} - 특정 세션 조회
특정 세션의 정보를 반환합니다.

- **Endpoint:** `GET /sessions/{session_id}`
- **Path Parameter:** `session_id` (정수) - 조회할 세션 ID
- **Response Body (`SessionResponse`):**
    ```json
    {
      "id": 1,
      "started_at": "2026-01-30T10:00:00.123456",
      "ended_at": "2026-01-30T11:30:00.654321"
    }
    ```
- **Error Response:**
    - `404 Not Found`: 세션이 존재하지 않는 경우
    ```json
    {"detail": "Session 999 not found"}
    ```

---

## 3. 데이터 모델 요약 (Schemas)

검사 세션의 생성, 종료, 조회 기능을 제공합니다.

### sessions 테이블
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,          -- 세션 시작 시간 (ISO 8601)
    ended_at TEXT                      -- 세션 종료 시간 (ISO 8601, NULL=진행중)
)
```

### inspection_logs 테이블
```sql
CREATE TABLE inspection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601 형식
    image_id TEXT NOT NULL,            -- 이미지 고유 ID
    result TEXT NOT NULL,              -- 'defect' 또는 'normal'
    detections TEXT,                   -- JSON 배열: [{"defect_type": "", "confidence": 0.0, "bbox": []}]
    image_path TEXT,                   -- 이미지 저장 경로 (nullable)
    session_id INTEGER,                -- 세션 ID (FK, nullable)
    FOREIGN KEY (session_id) REFERENCES sessions(id)
)
```

| 모델명 | 분류 | 용도 | 주요 필드 |
| :--- | :--- | :--- | :--- |
| **DetectRequest** | Request | POST /detect 요청 | `timestamp`, `image_id`, `image`, `detections[]`, `session_id` |
| **Detection** | Embedded | 개별 결함 정보 | `defect_type`, `confidence`, `bbox` |
| **DetectResponse** | Response | POST /detect 응답 | `status`, `id` |
| **InspectionLogResponse** | Response | GET /latest, Stats의 last_defect | `id`, `timestamp`, `image_id`, `result`, `detections[]`, `image_path`, `session_id` |
| **StatsResponse** | Response | GET /stats 응답 | `total_inspections`, `normal_count`, `defect_items`, `total_defects`, `defect_rate`, `avg_defects_per_item`, `avg_fps`, `last_defect` |
| **SessionResponse** | Response | 세션 정보 응답 | `id`, `started_at`, `ended_at` |
| **SessionCreateResponse** | Response | POST /sessions 응답 | `id`, `started_at` |
| **SessionListResponse** | Response | GET /sessions 응답 | `sessions[]` |

---

### 3.7 알림 조회 (Alerts)

### 4.1 이미지 저장 방식
- **저장 조건**: `image` 데이터가 있는 경우 서버의 `images/defects/` 디렉토리에 저장 (정상/불량 관계없이)
- **파일명 형식**: `{timestamp}_{image_id}.jpg`
  - 예: `2026-01-25T04-12-52.046352_09_spur_08.jpg.jpg`
  - 타임스탬프의 `:` 문자는 `-`로 치환됨 (파일 시스템 호환성)
- **image_path 생성**: 저장된 이미지 경로가 데이터베이스에 기록됨
- **이미지 미전송 시**: `image` 필드가 없거나 null인 경우 `image_path=null`

---

### 3.8 피드백 시스템 (Feedback)

품질 관리자가 검사 결과를 검토하고 피드백을 제공하는 MLOps 시스템입니다. 피드백 데이터는 자동으로 S3에 저장되어 모델 재학습에 활용됩니다.

#### 3.8.1 다중 bbox 피드백 + 자동 재라벨링

1개 PCB(log_id)에 대해 여러 bbox 피드백을 제출하고, 즉시 S3 refined/ 폴더에 학습 데이터를 저장합니다.

- **Endpoint:** `POST /feedback/bulk`
- **Request Body (`BulkFeedbackRequest`):**
    ```json
    {
      "log_id": 42,
      "image_width": 1200,
      "image_height": 760,
      "feedbacks": [
        {
          "target_bbox": [50, 60, 70, 80],
          "feedback_type": "false_positive",
          "comment": "먼지 오탐"
        },
        {
          "target_bbox": [100, 110, 120, 130],
          "feedback_type": "tp_wrong_class",
          "correct_label": "scratch",
          "comment": "hole이 아니라 scratch"
        }
      ],
      "false_negative_memo": "좌측 하단 scratch 누락",
      "created_by": "qa_team"
    }
    ```

**처리 플로우:**
1. 원본 detections 조회
2. 각 bbox에 대해 피드백 매칭 (target_bbox 좌표 비교)
3. 최종 라벨 생성:
   - 피드백 없음 → 원본 유지 (암묵적 TP)
   - `false_positive` → 라벨에서 삭제
   - `tp_wrong_class` → 클래스 수정
4. S3 refined/ 폴더에 이미지 복사 + YOLO 라벨 저장
5. DB에 피드백 저장 + `is_verified = true` 설정
6. FALSE_NEGATIVE가 있으면 needs_labeling/ 폴더에 복사

- **Response Body (`BulkFeedbackResponse`):** (201 Created)
    ```json
    {
      "status": "ok",
      "log_id": 42,
      "feedback_ids": [1, 2, 3],
      "saved_to_s3": true,
      "refined_path": "refined/20260130/42_pcb001.jpg",
      "final_label_count": 3,
      "false_negative_pending": true,
      "created_at": "2026-01-30T10:30:45"
    }
    ```

**에러 응답:**
- 400 Bad Request: image_path 없음
- 404 Not Found: 존재하지 않는 log_id
- 500 Internal Server Error: S3 저장 실패

---

#### 3.8.2 피드백 통계 조회 (bbox 기반 정확도 분석)

MLOps 모니터링용 정확도 분석 API입니다. 검증된 defect의 bbox별 정확도를 계산합니다.

- **Endpoint:** `GET /feedback/stats`
- **Query Parameters:**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `session_id` | string | X | 세션 필터 (생략: 전체, `latest`: 활성 세션, 숫자: 특정 세션) |

- **사용 예시:**
    ```bash
    # 전체 통계
    GET /feedback/stats

    # 최신 활성 세션만
    GET /feedback/stats?session_id=latest

    # 특정 세션
    GET /feedback/stats?session_id=1
    ```

- **Response Body (`FeedbackStatsResponse`):**
    ```json
    {
      "image_stats": {
        "total": 100,
        "by_result": {"defect": 15, "normal": 85},
        "verified": 50,
        "unverified": 50,
        "verification_rate": 50.0,
        "verified_by_result": {"defect": 10, "normal": 40}
      },
      "bbox_stats": {
        "total": 25,
        "correct": 20,
        "false_positive": 3,
        "wrong_class": 2,
        "accuracy_rate": 80.0,
        "by_defect_type": {
          "scratch": {"total": 15, "correct": 12, "fp": 2, "wrong": 1, "accuracy": 80.0},
          "hole": {"total": 10, "correct": 8, "fp": 1, "wrong": 1, "accuracy": 80.0}
        }
      },
      "feedback_stats": {
        "total": 8,
        "false_positive": 3,
        "tp_wrong_class": 2,
        "false_negative": 3
      },
      "class_confusion": [
        {"from_class": "hole", "to_class": "scratch", "count": 2}
      ]
    }
    ```

**응답 필드 설명:**

| 필드 | 설명 |
|------|------|
| `image_stats` | 이미지 단위 통계 (전체, 검증률) |
| `bbox_stats` | bbox 단위 정확도 (검증된 defect만, **핵심 지표**) |
| `feedback_stats` | 피드백 타입별 집계 |
| `class_confusion` | 클래스 혼동 패턴 (어떤 클래스를 어떤 클래스로 잘못 예측) |

**에러 응답:**
- 404 Not Found: 존재하지 않는 session_id

---

#### 3.8.3 라벨링 대기열 조회

처리되지 않은 피드백 목록을 반환합니다. 라벨링 도구에서 사용합니다.

- **Endpoint:** `GET /feedback/queue`
- **Query Parameters:**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `session_id` | string | X | 세션 필터 (생략: 전체, `latest`: 활성 세션, 숫자: 특정 세션) |

- **Response Body (`List[FeedbackQueueResponse]`):**
    ```json
    [
      {
        "feedback_id": 1,
        "log_id": 42,
        "image_url": "https://s3...presigned-url...",
        "feedback_type": "false_positive",
        "comment": "먼지 오탐",
        "created_at": "2026-01-30T10:30:45",
        "original_detections": [
          {"defect_type": "scratch", "confidence": 0.95, "bbox": [10, 20, 100, 120]}
        ],
        "target_bbox": [10, 20, 100, 120]
      }
    ]
    ```

---

#### 3.8.4 데이터셋 내보내기 정보

MLOps 파이프라인 연동용 데이터셋 정보를 반환합니다.

- **Endpoint:** `GET /feedback/export`
- **Response Body:**
    ```json
    {
      "status": "ready",
      "dataset_info": {
        "s3_uri": "s3://pcb-data-storage/refined/",
        "bucket": "pcb-data-storage",
        "prefix": "refined/",
        "image_count": 150
      },
      "command_guide": "aws s3 sync s3://pcb-data-storage/refined/ ./dataset/refined/",
      "message": "Use the s3_uri to sync your training data."
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
| feedback_type | TEXT | 피드백 종류 (false_positive, false_negative, tp_wrong_class) |
| correct_label | TEXT | 올바른 라벨 (tp_wrong_class 시 사용) |
| target_bbox | TEXT | 대상 bbox JSON ([x1, y1, x2, y2]) |
| comment | TEXT | 추가 설명 (nullable) |
| created_at | TEXT | 생성 시각 (ISO 8601) |
| created_by | TEXT | 작성자 (nullable) |
| status | TEXT | 상태 (pending, resolved) DEFAULT 'pending' |

### 4.4 inspection_logs 추가 컬럼 (검증 관련)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| is_verified | BOOLEAN | 검증 완료 여부 DEFAULT FALSE |
| verified_at | TEXT | 검증 완료 시각 (ISO 8601) |
| verified_by | TEXT | 검증자 (nullable) |

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
