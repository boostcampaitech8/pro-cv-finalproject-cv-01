# Pro-CV API Data Format Specification

본 문서는 `pro-cv` 프로젝트의 백엔드(`serving/api`)가 프론트엔드 및 엣지 디바이스와 주고받는 데이터의 형태를 분석한 결과입니다.

## 1. 개요
백엔드는 **FastAPI**로 구현되어 있으며, 데이터 검증을 위해 **Pydantic** 모델(Schemas)을 사용합니다. 주요 통신 대상은 검사 결과를 전송하는 **Edge/Jetson** 기기와 이를 시각화하는 **Streamlit** 프론트엔드입니다.

---

## 2. API 상세 데이터 포맷

### 2.1 검사 결과 수신 (Detect)
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
          "defect_type": "string", // 결함 종류 (scratch, dent, hole 등)
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
      "id": 123                    // 데이터베이스에 저장된 첫 번째 로그의 ID
    }
    ```

---

### 2.2 통계 정보 조회 (Stats)
프론트엔드 대시보드 구성을 위한 요약 통계 정보를 제공합니다.

- **Endpoint:** `GET /stats`
- **Query Parameter:** `session_id` (선택) - 특정 세션의 통계만 조회. 생략 시 전체 통계 반환.
- **Response Body (`StatsResponse`):**
    ```json
    {
      "total_inspections": 100,     // 총 검사 건수 (이미지 기준)
      "normal_count": 85,           // 정상 판정건수
      "defect_items": 15,           // 불량 판정건수 (이미지 기준)
      "total_defects": 22,          // 탐지된 총 결함 개수 (모든 이미지의 결함 합계)
      "defect_rate": 15.0,          // 불량률 (%)
      "avg_defects_per_item": 1.47, // 불량 이미지당 평균 결함 개수
      "avg_fps": 0.0,               // 평균 처리 FPS (현재 미구현)
      "last_defect": {              // 가장 최근 발생한 불량 이미지 정보 (또는 null)
        "id": 123,
        "timestamp": "2026-01-25T04:12:52.046352",
        "image_id": "09_spur_08.jpg",
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

### 2.3 최근 로그 목록 (Latest Logs)
검사 이력을 리스트 형태로 조회합니다.

- **Endpoint:** `GET /latest`
- **Query Parameters:**
    - `limit` (기본값: 10) - 반환할 로그 개수
    - `session_id` (선택) - 특정 세션의 로그만 조회. 생략 시 전체 로그 반환.
- **Response Body (List of InspectionLogResponse):**
    ```json
    [
      {
        "id": 10,
        "timestamp": "2026-01-25T04:12:52.046352",
        "image_id": "09_spur_08.jpg",
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
      },
      ...
    ]
    ```

---

### 2.4 결함 유형별 집계 (Defect Aggregation)
각 결함 종류별로 발생 횟수를 집계하여 반환합니다.

- **Endpoint:** `GET /defects`
- **Query Parameter:** `session_id` (선택) - 특정 세션의 결함만 집계. 생략 시 전체 결함 집계.
- **Response Body:**
    ```json
    {
      "scratch": 12,
      "dent": 5,
      "hole": 3,
      "unknown": 2
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

## 4. 특이 사항

### 4.1 이미지 저장 방식
- **저장 조건**: `image` 데이터가 있는 경우 서버의 `images/defects/` 디렉토리에 저장 (정상/불량 관계없이)
- **파일명 형식**: `{timestamp}_{image_id}.jpg`
  - 예: `2026-01-25T04-12-52.046352_09_spur_08.jpg.jpg`
  - 타임스탬프의 `:` 문자는 `-`로 치환됨 (파일 시스템 호환성)
- **image_path 생성**: 저장된 이미지 경로가 데이터베이스에 기록됨
- **이미지 미전송 시**: `image` 필드가 없거나 null인 경우 `image_path=null`

---

### 4.2 다중 결함 처리 (Schema v2.0)

#### ✅ 현재 (v2.0 - BE 브랜치)
- **1행 = 1이미지**: 각 검사 결과는 데이터베이스에 **정확히 1개의 행**으로 저장됨
- **모든 결함 통합**: 하나의 이미지에서 여러 결함이 탐지되면 모두 `detections` JSON 배열에 포함됨
- **예시**:
  ```json
  // 입력: 4개의 결함 탐지
  POST /detect
  {
    "image_id": "09_spur_08.jpg",
    "detections": [
      {"defect_type": "defect_5", "confidence": 0.8187, "bbox": [627, 542, 724, 612]},
      {"defect_type": "defect_5", "confidence": 0.7866, "bbox": [859, 1114, 911, 1173]},
      {"defect_type": "defect_5", "confidence": 0.7827, "bbox": [1396, 915, 1500, 961]},
      {"defect_type": "defect_5", "confidence": 0.5699, "bbox": [2547, 783, 2634, 849]}
    ]
  }
  
  // DB 저장: 1행만 생성
  id | timestamp | image_id | result | detections (JSON) | image_path
  69 | 2026-01-25T04:12:52.046352 | 09_spur_08.jpg | defect | [4개 결함 배열] | images/defects/...
