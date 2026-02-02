# POST /feedback API 문서

## 개요

품질 관리자가 검사 결과를 검토하고 피드백을 제공할 수 있는 API입니다.

## 엔드포인트

**POST /feedback/**

검사 결과에 대한 피드백 생성

## 피드백 종류

| 타입 | 설명 | correct_label 필요 |
|------|------|-------------------|
| `false_positive` | 정상인데 불량으로 오탐 | X |
| `false_negative` | 불량인데 정상으로 통과 | X |
| `label_correction` | 결함은 맞지만 종류가 틀림 | O (필수) |

## Request Body

```json
{
  "log_id": 42,                           // 검사 로그 ID (필수)
  "feedback_type": "label_correction",    // 피드백 종류 (필수)
  "correct_label": "scratch",             // 올바른 라벨 (label_correction 시 필수)
  "comment": "추가 설명",                  // 선택사항 (최대 500자)
  "created_by": "manager_kim"             // 작성자 (선택사항, 최대 100자)
}
```

### 필드 설명

| 필드 | 타입 | 필수 | 제약사항 |
|------|------|------|---------|
| log_id | integer | O | 양수, inspection_logs에 존재해야 함 |
| feedback_type | string | O | false_positive, false_negative, label_correction 중 하나 |
| correct_label | string | △ | label_correction 시 필수, scratch/hole/contamination/crack/normal 중 하나 |
| comment | string | X | 최대 500자 |
| created_by | string | X | 최대 100자 |

## Response (201 Created)

```json
{
  "id": 1,
  "log_id": 42,
  "feedback_type": "label_correction",
  "correct_label": "scratch",
  "comment": "추가 설명",
  "created_at": "2026-01-30T10:30:45.123456",
  "created_by": "manager_kim",
  "status": "ok"
}
```

## 에러 응답

### 404 Not Found - 존재하지 않는 log_id

```json
{
  "detail": "Inspection log 9999 not found"
}
```

### 422 Validation Error - correct_label 누락

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "correct_label required for label_correction"
    }
  ]
}
```

### 422 Validation Error - correct_label 허용값 외

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "correct_label must be one of {'scratch', 'hole', 'contamination', 'crack', 'normal'}, got 'unknown_defect'"
    }
  ]
}
```

### 422 Validation Error - 잘못된 feedback_type

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "feedback_type"],
      "msg": "feedback_type must be one of {'false_positive', 'false_negative', 'label_correction'}"
    }
  ]
}
```

## 사용 예시

### 1. False Positive (오탐)

```bash
curl -X POST http://localhost:8000/feedback/ \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": 42,
    "feedback_type": "false_positive",
    "comment": "정상 PCB인데 먼지로 인해 오탐됨",
    "created_by": "manager_kim"
  }'
```

### 2. False Negative (미탐)

```bash
curl -X POST http://localhost:8000/feedback/ \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": 43,
    "feedback_type": "false_negative",
    "comment": "불량인데 정상으로 통과됨",
    "created_by": "manager_lee"
  }'
```

### 3. Label Correction (라벨 수정)

```bash
curl -X POST http://localhost:8000/feedback/ \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": 44,
    "feedback_type": "label_correction",
    "correct_label": "scratch",
    "comment": "hole이 아니라 scratch입니다",
    "created_by": "qa_team"
  }'
```

## 테스트

### 자동 테스트 실행

```bash
# 서버 시작
uv run uvicorn main:app --reload --port 8000

# 테스트 실행 (다른 터미널)
uv run python test/test_feedback_api.py
```

### Swagger UI 테스트

http://localhost:8000/docs 에서 대화형 테스트 가능

## 데이터베이스

### 테이블 스키마

```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL,
    correct_label TEXT,
    comment TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    FOREIGN KEY (log_id) REFERENCES inspection_logs(id) ON DELETE CASCADE
);
```

### 인덱스

- `idx_feedback_log_id` - log_id 기준 조회 최적화
- `idx_feedback_type` - feedback_type 기준 조회 최적화
- `idx_feedback_created_at` - 시간순 정렬 최적화

### 데이터 조회

```bash
# 모든 피드백 조회
sqlite3 data/inspection.db "SELECT * FROM feedback ORDER BY created_at DESC;"

# 특정 로그의 피드백 조회
sqlite3 data/inspection.db "SELECT * FROM feedback WHERE log_id = 42;"

# 피드백 타입별 집계
sqlite3 data/inspection.db "SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type;"
```

## 향후 확장 (구현 범위 외)

1. **GET /feedback/{log_id}** - 특정 검사에 대한 피드백 조회
2. **GET /feedback** - 모든 피드백 조회 (필터 옵션)
3. **GET /feedback/stats** - 피드백 통계 (타입별, 세션별 집계)
4. **재라벨링 큐** - 피드백 데이터를 모델 재학습에 활용

## 구현 파일

| 파일 | 역할 |
|------|------|
| `config/settings.py` | 허용값 상수 (ALLOWED_DEFECT_TYPES, ALLOWED_FEEDBACK_LABELS) |
| `database/db.py` | DB 함수 (log_exists, add_feedback, get_feedback_by_log_id) |
| `schemas/schemas.py` | Pydantic 스키마 (FeedbackRequest, FeedbackResponse) |
| `routers/feedback.py` | POST /feedback 엔드포인트 |
| `main.py` | 라우터 등록 |
| `test/test_feedback_api.py` | 통합 테스트 |
