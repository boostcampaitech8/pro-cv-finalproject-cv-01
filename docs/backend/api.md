# API 명세

백엔드 API 엔드포인트 상세 명세입니다.

## API 엔드포인트 목록

| 카테고리 | Method | Endpoint | 설명 |
|---------|--------|----------|------|
| Detect | POST | `/detect/` | 검사 결과 수신 |
| Stats | GET | `/stats` | 통계 정보 조회 |
| Stats | GET | `/latest` | 최근 검사 로그 |
| Stats | GET | `/defects` | 결함 타입별 집계 |
| Sessions | POST | `/sessions/` | 세션 생성 |
| Sessions | PATCH | `/sessions/{id}` | 세션 종료 |
| Sessions | GET | `/sessions/` | 세션 목록 조회 |
| Sessions | GET | `/sessions/{id}` | 특정 세션 조회 |
| Monitoring | GET | `/monitoring/health` | 시스템 건강 상태 |
| Monitoring | GET | `/monitoring/alerts` | 알림 조회 |
| Feedback | POST | `/feedback/bulk` | 다중 bbox 피드백 |
| Feedback | GET | `/feedback/stats` | 피드백 통계 |
| Feedback | GET | `/feedback/queue` | 라벨링 대기열 |
| Feedback | GET | `/feedback/export` | 데이터셋 내보내기 |
| Images | GET | `/raw/{path}` | 이미지 조회 |

## POST /detect/ (엣지 → 백엔드)

검사 결과 수신. 1개 PCB에서 여러 결함 지원.

**Request Body:**

```json
{
  "timestamp": "2026-01-18T15:01:00",
  "image_id": "PCB_002",
  "image": "base64_encoded_string",
  "session_id": 1,
  "detections": [
    {
      "defect_type": "scratch",
      "confidence": 0.95,
      "bbox": [10, 20, 100, 120]
    }
  ]
}
```

**Response:**

```json
{
  "status": "ok",
  "id": 1
}
```

## GET /stats (프론트 → 백엔드)

통계 정보 조회.

**Query Parameters:**
- `session_id` (선택): 특정 세션의 통계만 조회

**Response:**

```json
{
  "total_inspections": 100,
  "normal_count": 85,
  "defect_items": 15,
  "total_defects": 23,
  "defect_rate": 15.0,
  "avg_defects_per_item": 1.53,
  "avg_fps": 0.0,
  "last_defect": {
    "id": 123,
    "timestamp": "2026-01-25T04:12:52",
    "image_id": "PCB_001",
    "result": "defect",
    "detections": [...],
    "image_path": "...",
    "session_id": 1
  }
}
```

## GET /latest

최근 검사 이력 조회.

**Query Parameters:**
- `limit` (기본값: 10): 반환할 로그 개수
- `session_id` (선택): 특정 세션의 로그만 조회

**Response:**

```json
[
  {
    "id": 123,
    "timestamp": "2026-01-25T04:12:52",
    "image_id": "PCB_001",
    "result": "defect",
    "detections": [...],
    "image_path": "...",
    "session_id": 1
  }
]
```

## GET /defects

결함 타입별 집계.

**Query Parameters:**
- `session_id` (선택): 특정 세션의 결함만 집계

**Response:**

```json
{
  "scratch": 15,
  "hole": 8,
  "contamination": 3,
  "crack": 2
}
```

## 세션 관리 API

### POST /sessions/ - 세션 생성

**Response:**

```json
{
  "id": 1,
  "started_at": "2026-01-30T10:00:00.123456"
}
```

### PATCH /sessions/{session_id} - 세션 종료

**Response:**

```json
{
  "id": 1,
  "started_at": "2026-01-30T10:00:00.123456",
  "ended_at": "2026-01-30T11:30:00.654321"
}
```

### GET /sessions/ - 세션 목록 조회

**Response:**

```json
{
  "sessions": [
    {
      "id": 3,
      "started_at": "2026-01-30T14:00:00.000000",
      "ended_at": null
    }
  ]
}
```

## GET /monitoring/health

시스템 건강 상태 조회.

**Query Parameters:**
- `session_id` (선택): 특정 세션의 상태만 조회

**Response:**

```json
{
  "status": "warning",
  "timestamp": "2026-01-29T15:30:00",
  "session_info": {
    "id": 1,
    "started_at": "2026-01-29T10:00:00",
    "ended_at": null,
    "duration_seconds": 19800.0,
    "is_active": true
  },
  "total_inspections": 100,
  "normal_count": 85,
  "defect_count": 15,
  "defect_rate": 15.0,
  "total_defects": 23,
  "avg_defects_per_item": 1.53,
  "defect_confidence_stats": {
    "avg_confidence": 0.82,
    "min_confidence": 0.55,
    "max_confidence": 0.98,
    "distribution": {
      "high": 70,
      "medium": 20,
      "low": 8,
      "very_low": 2
    }
  },
  "defect_type_stats": [
    {
      "defect_type": "scratch",
      "count": 15,
      "avg_confidence": 0.88
    }
  ],
  "alerts": [
    {
      "level": "warning",
      "message": "불량률이 15.0%로 높습니다",
      "value": 15.0,
      "threshold": 10.0,
      "action": "품질 모니터링 강화 권장"
    }
  ]
}
```

## GET /monitoring/alerts

경량 알림 API.

**Query Parameters:**
- `session_id` (선택): 특정 세션의 알림만 조회

**Response:**

```json
{
  "status": "critical",
  "timestamp": "2026-01-29T15:30:00",
  "session_info": {...},
  "alerts": [...],
  "summary": {
    "defect_rate": 25.0,
    "avg_confidence": 0.72
  }
}
```

## POST /feedback/bulk

다중 bbox 피드백 제출.

**Request Body:**

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

**Response:**

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

## GET /feedback/stats

피드백 통계 조회.

**Query Parameters:**
- `session_id` (선택): 특정 세션의 통계만 조회

**Response:**

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
      "scratch": {
        "total": 15,
        "correct": 12,
        "fp": 2,
        "wrong": 1,
        "accuracy": 80.0
      }
    }
  },
  "feedback_stats": {
    "total": 8,
    "false_positive": 3,
    "tp_wrong_class": 2,
    "false_negative": 3
  },
  "class_confusion": [
    {
      "from_class": "hole",
      "to_class": "scratch",
      "count": 2
    }
  ]
}
```

## GET /feedback/export

데이터셋 내보내기 정보.

**Response:**

```json
{
  "status": "ready",
  "dataset_info": {
    "s3_uri": "s3://pcb-data-storage/refined/",
    "bucket": "pcb-data-storage",
    "prefix": "refined/",
    "image_count": 150
  },
  "command_guide": "aws s3 sync s3://pcb-data-storage/refined/ ./dataset/refined/"
}
```

## GET /raw/{file_path}

S3 이미지 조회 (Presigned URL).

**Example:**
```
GET /raw/20260125/timestamp_PCB_001.jpg
```

**Response:**
307 Temporary Redirect → S3 Presigned URL
