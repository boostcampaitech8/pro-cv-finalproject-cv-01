# Feedback API 문서

## 개요

품질 관리자가 PCB 검사 결과를 검토하고 피드백을 제공하는 API입니다.
피드백 제출 시 S3에 학습 데이터(refined/)를 자동 저장하여 MLOps 재학습 파이프라인과 연동됩니다.

### 피드백 종류

| 타입 | 설명 | correct_label | target_bbox |
|------|------|:---:|:---:|
| `false_positive` | 정상인데 불량으로 오탐 | X | O (필수) |
| `tp_wrong_class` | 결함은 맞지만 종류가 틀림 | O (필수) | O (필수) |
| `false_negative` | 불량인데 정상으로 통과 (누락) | X | X |

### 허용 라벨 값

`correct_label`에 사용 가능한 값:

| 값 | YOLO 클래스 ID |
|------|:---:|
| `scratch` | 0 |
| `hole` | 1 |
| `contamination` | 2 |
| `crack` | 3 |
| `normal` | - |

---

## 엔드포인트 목록

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/feedback/bulk` | 다중 bbox 피드백 생성 + S3 자동 저장 |
| GET | `/feedback/stats` | 피드백 통계 조회 (bbox 기반 정확도 분석) |
| GET | `/feedback/queue` | 라벨링 대기열 조회 |
| GET | `/feedback/export` | MLOps 파이프라인용 데이터셋 정보 |

---

## POST /feedback/bulk

다중 bbox 피드백 생성 + 자동 S3 데이터셋 저장.

1개 PCB(`log_id`)에 대해 여러 bbox의 피드백을 한 번에 제출하고, 즉시 S3 `refined/` 폴더에 학습 데이터를 저장합니다.

### 처리 플로우

```
1. inspection_logs 조회 (원본 detections)
2. 각 bbox에 대해 피드백 찾기 (target_bbox 좌표 매칭)
3. 최종 라벨 생성
   - 피드백 없음 → 원본 유지 (암묵적 TP)
   - tp_wrong_class → 클래스 수정
   - false_positive → 라벨에서 삭제
4. S3 refined/ 저장 (이미지 복사 + YOLO 라벨 생성)
5. DB에 피드백 저장 (모든 피드백, FN 포함)
6. false_negative가 있으면 → S3 needs_labeling/ 복사
7. inspection_logs.is_verified = true 표시
```

### Request Body

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
    },
    {
      "feedback_type": "false_negative",
      "comment": "좌측 하단 scratch 누락"
    },
    {
      "feedback_type": "false_negative",
      "comment": "우측 상단 hole 누락"
    }
  ],
  "created_by": "qa_team"
}
```

#### 필드 설명

| 필드 | 타입 | 필수 | 제약사항 |
|------|------|:---:|---------|
| log_id | integer | O | 양수, `inspection_logs`에 존재해야 함 |
| image_width | integer | O | 양수, 크롭된 이미지 너비 (픽셀) |
| image_height | integer | O | 양수, 크롭된 이미지 높이 (픽셀) |
| feedbacks | FeedbackItem[] | O | 최대 100개, 빈 배열 허용 (모든 bbox 정답 = 암묵적 TP) |
| created_by | string | X | 최대 100자 |

#### FeedbackItem 필드

| 필드 | 타입 | 필수 | 제약사항 |
|------|------|:---:|---------|
| feedback_type | string | O | `false_positive`, `tp_wrong_class`, `false_negative` 중 하나 |
| target_bbox | list[int] | 조건부 | `false_positive`, `tp_wrong_class` 시 필수. `[x1, y1, x2, y2]`, x1 < x2, y1 < y2 |
| correct_label | string | 조건부 | `tp_wrong_class` 시 필수. `scratch`, `hole`, `contamination`, `crack`, `normal` 중 하나 |
| comment | string | X | 최대 500자 |

### Response (201 Created)

```json
{
  "status": "ok",
  "log_id": 42,
  "feedback_ids": [101, 102, 103, 104],
  "saved_to_s3": true,
  "refined_path": "refined/images/42_defect_PCB002.jpg",
  "final_label_count": 2,
  "false_negative_count": 2,
  "created_at": "2026-02-03T10:30:00.123456"
}
```

#### 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 처리 결과 (`"ok"`) |
| log_id | int | 대상 검사 로그 ID |
| feedback_ids | list[int] | 생성된 피드백 ID 목록 |
| saved_to_s3 | bool | S3 `refined/` 폴더 저장 성공 여부 |
| refined_path | string\|null | S3 refined 이미지 경로 (예: `refined/images/42_xxx.jpg`) |
| final_label_count | int | FP 제거 + 클래스 수정 후 최종 라벨 개수 |
| false_negative_count | int | `false_negative` 피드백 개수 |
| created_at | string | 피드백 생성 시각 (ISO 8601) |

### 에러 응답

#### 404 Not Found - 존재하지 않는 log_id

```json
{
  "detail": "Inspection log 9999 not found"
}
```

#### 400 Bad Request - 이미지 경로 없음

```json
{
  "detail": "Log 42 has no image_path"
}
```

#### 422 Validation Error - feedback_type 허용값 외

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "feedbacks", 0, "feedback_type"],
      "msg": "Value error, feedback_type must be one of {'false_positive', 'false_negative', 'tp_wrong_class'}"
    }
  ]
}
```

#### 422 Validation Error - correct_label 누락 (tp_wrong_class)

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "feedbacks", 0],
      "msg": "Value error, correct_label required for tp_wrong_class"
    }
  ]
}
```

#### 422 Validation Error - correct_label 허용값 외

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "feedbacks", 0],
      "msg": "Value error, correct_label must be one of {'scratch', 'hole', 'contamination', 'crack', 'normal'}"
    }
  ]
}
```

#### 422 Validation Error - target_bbox 누락 (false_positive/tp_wrong_class)

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "feedbacks", 0],
      "msg": "Value error, target_bbox required for false_positive"
    }
  ]
}
```

#### 422 Validation Error - target_bbox 형식 오류

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "feedbacks", 0],
      "msg": "Value error, target_bbox must have 4 elements [x1, y1, x2, y2]"
    }
  ]
}
```

#### 422 Validation Error - target_bbox 좌표 오류 (x1 >= x2)

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "feedbacks", 0],
      "msg": "Value error, x1 must be less than x2"
    }
  ]
}
```

#### 422 Validation Error - feedbacks 개수 초과

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "feedbacks"],
      "msg": "Value error, feedbacks cannot exceed 100 items, got 150"
    }
  ]
}
```

#### 500 Internal Server Error - S3 저장 실패

```json
{
  "detail": "Failed to save to S3: ..."
}
```

### 사용 예시

#### 1. 모든 bbox가 정답 (암묵적 TP)

```bash
curl -X POST http://localhost:8000/feedback/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": 42,
    "image_width": 1200,
    "image_height": 760,
    "feedbacks": [],
    "created_by": "manager_kim"
  }'
```

#### 2. False Positive (오탐 제거)

```bash
curl -X POST http://localhost:8000/feedback/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": 42,
    "image_width": 1200,
    "image_height": 760,
    "feedbacks": [
      {
        "target_bbox": [50, 60, 70, 80],
        "feedback_type": "false_positive",
        "comment": "정상 PCB인데 먼지로 인해 오탐됨"
      }
    ],
    "created_by": "manager_kim"
  }'
```

#### 3. 클래스 수정 (tp_wrong_class)

```bash
curl -X POST http://localhost:8000/feedback/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": 44,
    "image_width": 1200,
    "image_height": 760,
    "feedbacks": [
      {
        "target_bbox": [100, 110, 120, 130],
        "feedback_type": "tp_wrong_class",
        "correct_label": "scratch",
        "comment": "hole이 아니라 scratch입니다"
      }
    ],
    "created_by": "qa_team"
  }'
```

#### 4. 미탐 신고 (false_negative)

```bash
curl -X POST http://localhost:8000/feedback/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": 43,
    "image_width": 1200,
    "image_height": 760,
    "feedbacks": [
      {
        "feedback_type": "false_negative",
        "comment": "좌측 하단 scratch 누락"
      },
      {
        "feedback_type": "false_negative",
        "comment": "우측 상단 hole 누락"
      }
    ],
    "created_by": "manager_lee"
  }'
```

#### 5. 복합 피드백 (FP + 클래스 수정 + FN 동시)

```bash
curl -X POST http://localhost:8000/feedback/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": 45,
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
      },
      {
        "feedback_type": "false_negative",
        "comment": "좌측 하단 crack 누락"
      }
    ],
    "created_by": "qa_team"
  }'
```

---

## GET /feedback/stats

피드백 통계 조회. bbox 기반 모델 정확도를 분석하여 MLOps 모니터링에 활용합니다.

### Query Parameters

| 필드 | 타입 | 필수 | 설명 |
|------|------|:---:|------|
| session_id | string | X | 세션 필터. 생략 시 전체, `"latest"` 시 최신 세션, 숫자 시 해당 세션 |

### 사용 예시

```bash
# 전체 통계
GET /feedback/stats

# 최신 세션만
GET /feedback/stats?session_id=latest

# 특정 세션
GET /feedback/stats?session_id=3
```

### Response (200 OK)

```json
{
  "image_stats": {
    "total": 100,
    "by_result": { "defect": 30, "normal": 70 },
    "verified": 25,
    "unverified": 75,
    "verification_rate": 25.0,
    "verified_by_result": { "defect": 20, "normal": 5 }
  },
  "bbox_stats": {
    "total": 50,
    "correct": 42,
    "false_positive": 5,
    "wrong_class": 3,
    "accuracy_rate": 84.0,
    "by_defect_type": {
      "scratch": {
        "total": 30,
        "correct": 26,
        "fp": 3,
        "wrong": 1,
        "accuracy": 86.7
      },
      "hole": {
        "total": 20,
        "correct": 16,
        "fp": 2,
        "wrong": 2,
        "accuracy": 80.0
      }
    }
  },
  "feedback_stats": {
    "total": 10,
    "false_positive": 5,
    "tp_wrong_class": 3,
    "false_negative": 2
  },
  "class_confusion": [
    { "from_class": "hole", "to_class": "scratch", "count": 2 },
    { "from_class": "scratch", "to_class": "crack", "count": 1 }
  ]
}
```

#### 응답 구조

**image_stats** - 이미지 단위 통계

| 필드 | 타입 | 설명 |
|------|------|------|
| total | int | 전체 이미지 개수 |
| by_result | dict | 결과별 개수 (`{"defect": N, "normal": N}`) |
| verified | int | 검증 완료 이미지 개수 |
| unverified | int | 미검증 이미지 개수 |
| verification_rate | float | 검증률 (%) |
| verified_by_result | dict | 검증된 이미지의 결과별 개수 |

**bbox_stats** - bbox 단위 정확도 (검증된 defect만)

| 필드 | 타입 | 설명 |
|------|------|------|
| total | int | 전체 bbox 개수 |
| correct | int | 정답 개수 (피드백 없음 = 암묵적 TP) |
| false_positive | int | 오탐 개수 |
| wrong_class | int | 클래스 오류 개수 |
| accuracy_rate | float | 정확도 (%) |
| by_defect_type | dict | 결함 타입별 정확도 (아래 참조) |

**by_defect_type[타입]** - 결함 타입별 정확도

| 필드 | 타입 | 설명 |
|------|------|------|
| total | int | 해당 타입 bbox 개수 |
| correct | int | 정답 개수 |
| fp | int | 오탐 개수 |
| wrong | int | 클래스 오류 개수 |
| accuracy | float | 정확도 (%) |

**feedback_stats** - 피드백 타입별 집계

| 필드 | 타입 | 설명 |
|------|------|------|
| total | int | 전체 피드백 개수 |
| false_positive | int | 오탐 피드백 개수 |
| tp_wrong_class | int | 클래스 오류 피드백 개수 |
| false_negative | int | 미탐 피드백 개수 |

**class_confusion** - 클래스 혼동 패턴 (count 내림차순)

| 필드 | 타입 | 설명 |
|------|------|------|
| from_class | string | 원본 예측 클래스 |
| to_class | string | 실제 정답 클래스 |
| count | int | 발생 횟수 |

### 에러 응답

#### 404 Not Found - 세션을 찾을 수 없음

```json
{
  "detail": "Session not found"
}
```

#### 500 Internal Server Error

```json
{
  "detail": "Failed to get feedback stats: ..."
}
```

---

## GET /feedback/queue

처리되지 않은(pending) 피드백 목록을 조회합니다. 라벨링 도구에서 대기열을 표시하는 데 사용됩니다.

### Query Parameters

| 필드 | 타입 | 필수 | 설명 |
|------|------|:---:|------|
| session_id | string | X | 세션 필터. 생략 시 전체, `"latest"` 시 최신 세션, 숫자 시 해당 세션 |

### 사용 예시

```bash
# 전체 대기열
GET /feedback/queue

# 최신 세션만
GET /feedback/queue?session_id=latest

# 특정 세션
GET /feedback/queue?session_id=3
```

### Response (200 OK)

`List[FeedbackQueueResponse]` 배열 반환 (생성 시각 내림차순)

```json
[
  {
    "feedback_id": 101,
    "log_id": 42,
    "image_url": "https://pcb-data-storage.s3.ap-southeast-2.amazonaws.com/raw/20260203/defect_PCB002.jpg?X-Amz-...",
    "feedback_type": "false_negative",
    "comment": "좌측 하단 scratch 누락",
    "created_at": "2026-02-03T10:30:00.123456",
    "original_detections": [
      { "defect_type": "scratch", "confidence": 0.95, "bbox": [10, 20, 100, 120] },
      { "defect_type": "hole", "confidence": 0.87, "bbox": [150, 180, 200, 230] }
    ],
    "target_bbox": null
  },
  {
    "feedback_id": 102,
    "log_id": 42,
    "image_url": "https://pcb-data-storage.s3...",
    "feedback_type": "false_positive",
    "comment": "먼지 오탐",
    "created_at": "2026-02-03T10:30:00.123456",
    "original_detections": [...],
    "target_bbox": [50, 60, 70, 80]
  }
]
```

#### 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| feedback_id | int | 피드백 ID |
| log_id | int | 대상 검사 로그 ID |
| image_url | string | S3 Presigned URL (1시간 유효) |
| feedback_type | string | 피드백 종류 |
| comment | string\|null | 사용자 코멘트 |
| created_at | string | 피드백 생성 시각 (ISO 8601) |
| original_detections | list[dict] | 원본 AI 예측 결과 (참고용) |
| target_bbox | list[int]\|null | 대상 bbox `[x1, y1, x2, y2]` |

### 에러 응답

#### 500 Internal Server Error

```json
{
  "detail": "Failed to get queue: ..."
}
```

---

## GET /feedback/export

MLOps 파이프라인 연동용 데이터셋 정보를 반환합니다. 학습 서버에서 이 API를 호출하여 데이터 위치와 개수를 확인합니다.

### 사용 예시

```bash
GET /feedback/export
```

### Response (200 OK)

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

#### 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 데이터셋 상태 (`"ready"`) |
| dataset_info.s3_uri | string | S3 URI (전체 경로) |
| dataset_info.bucket | string | S3 버킷 이름 |
| dataset_info.prefix | string | S3 prefix |
| dataset_info.image_count | int | refined 이미지 개수 |
| command_guide | string | `aws s3 sync` 사용 예시 |
| message | string | 안내 메시지 |

### 에러 응답

#### 500 Internal Server Error

```json
{
  "detail": "Failed to export dataset info: ..."
}
```

---

## 데이터베이스

### feedback 테이블

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL,        -- false_positive, tp_wrong_class, false_negative
    correct_label TEXT,                 -- tp_wrong_class 시 올바른 라벨
    comment TEXT,                       -- 추가 설명
    created_at TEXT,                    -- ISO 8601 생성 시각
    created_by TEXT,                    -- 작성자
    status TEXT DEFAULT 'pending',      -- pending, resolved
    target_bbox TEXT,                   -- JSON 문자열 [x1, y1, x2, y2]
    FOREIGN KEY (log_id) REFERENCES inspection_logs (id)
);
```

### 인덱스

| 인덱스 | 컬럼 | 용도 |
|--------|------|------|
| `idx_feedback_log_id` | log_id | log_id 기준 조회 최적화 |
| `idx_feedback_type` | feedback_type | 피드백 타입별 조회 최적화 |
| `idx_feedback_created_at` | created_at | 시간순 정렬 최적화 |

### inspection_logs 관련 컬럼 (피드백 시스템용)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| is_verified | BOOLEAN | 피드백 검증 완료 여부 (기본값: FALSE) |
| verified_at | TEXT | 검증 완료 시각 (ISO 8601) |
| verified_by | TEXT | 검증자 |

### 데이터 조회

```bash
# 모든 피드백 조회
sqlite3 data/inspection.db "SELECT * FROM feedback ORDER BY created_at DESC;"

# 특정 로그의 피드백 조회
sqlite3 data/inspection.db "SELECT * FROM feedback WHERE log_id = 42;"

# 피드백 타입별 집계
sqlite3 data/inspection.db "SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type;"

# 검증된 이미지 조회
sqlite3 data/inspection.db "SELECT id, image_id, verified_at, verified_by FROM inspection_logs WHERE is_verified = 1;"
```

---

## S3 데이터셋 자동 저장

### 폴더 구조

```
s3://pcb-data-storage/
├── raw/                      # 원본 이미지 (POST /detect 시 저장)
│   └── 20260203/
│       └── defect_PCB002.jpg
├── refined/                  # 재라벨링 완료 데이터 (POST /feedback/bulk 시 저장)
│   ├── images/
│   │   └── 42_defect_PCB002.jpg
│   └── labels/
│       └── 42_defect_PCB002.txt   # YOLO 포맷 라벨
└── needs_labeling/           # false_negative 이미지 (수동 라벨링 필요)
    ├── images/
    │   └── 43_defect_PCB003.jpg
    └── metadata/
        └── 43_defect_PCB003.json  # FN 코멘트 + 원본 예측 정보
```

### YOLO 라벨 포맷 (refined/labels/*.txt)

```
class_id x_center y_center width height
```

- 모든 값은 0.0~1.0 정규화 좌표
- 한 줄에 하나의 bbox
- `image_width`, `image_height`를 기준으로 정규화

예시:
```
0 0.050000 0.125000 0.016667 0.026316
3 0.183333 0.157895 0.016667 0.026316
```

### needs_labeling 메타데이터 (*.json)

```json
{
  "log_id": 43,
  "original_s3_key": "raw/20260203/defect_PCB003.jpg",
  "false_negative_comments": [
    "좌측 하단 scratch 누락",
    "우측 상단 hole 누락"
  ],
  "false_negative_count": 2,
  "original_detections": [
    { "defect_type": "scratch", "confidence": 0.95, "bbox": [10, 20, 100, 120] }
  ],
  "created_at": "2026-02-03T10:30:00.123456"
}
```

---

## 핵심 로직

### bbox 매칭 (좌표 비교)

피드백의 `target_bbox`와 원본 detection의 `bbox`를 비교할 때 **2픽셀 오차를 허용**합니다.

```
bbox_equals([50, 60, 70, 80], [51, 60, 70, 80]) → True  (1px 차이)
bbox_equals([50, 60, 70, 80], [55, 60, 70, 80]) → False (5px 차이)
```

### 최종 라벨 생성 규칙

| 원본 Detection | 매칭된 피드백 | 결과 |
|---------------|-------------|------|
| bbox A | 없음 | 원본 그대로 유지 (암묵적 TP) |
| bbox B | false_positive | 라벨에서 삭제 |
| bbox C | tp_wrong_class (→ scratch) | 클래스 ID를 scratch로 변경 |
| - | false_negative | 원본 라벨에 영향 없음, `needs_labeling/`에 별도 복사 |

---

## 구현 파일

| 파일 | 역할 |
|------|------|
| `routers/feedback.py` | 엔드포인트 4개 + 헬퍼 함수 (bbox_equals, normalize_bbox 등) |
| `schemas/schemas.py` | Pydantic 모델 (FeedbackItem, BulkFeedbackRequest, BulkFeedbackResponse, FeedbackStatsResponse 등) |
| `database/db.py` | DB 함수 (add_feedback, get_feedback_stats, get_feedback_queue, mark_as_verified 등) |
| `utils/s3_dataset.py` | S3 저장 (save_to_refined, copy_to_needs_labeling, get_refined_dataset_stats) |
| `utils/image_utils.py` | Presigned URL 생성 (generate_presigned_url) |
| `config/settings.py` | 허용값 상수 (ALLOWED_DEFECT_TYPES, ALLOWED_FEEDBACK_LABELS, CLASS_ID_MAP) |
| `main.py` | 라우터 등록 |

## 테스트

```bash
# 서버 시작
cd serving/api
uv run uvicorn main:app --reload --port 8000

# Swagger UI 테스트
# http://localhost:8000/docs

# 피드백 API 테스트
uv run python test/test_feedback_api.py
```
