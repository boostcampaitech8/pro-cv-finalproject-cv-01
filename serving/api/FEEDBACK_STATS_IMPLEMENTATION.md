# Feedback Stats 엔드포인트 구현 완료

## 구현 내용

### GET /feedback/stats - 피드백 통계 조회

MLOps 모니터링용 피드백 통계 API 구현 완료.

## 변경된 파일

### 1. `schemas/schemas.py` (3개 스키마 추가)

```python
class FeedbackTypeStats(BaseModel):
    """피드백 종류별 통계"""
    false_positive: int = Field(default=0)
    false_negative: int = Field(default=0)
    label_correction: int = Field(default=0)

class DefectTypeFeedbackStats(BaseModel):
    """결함 타입별 피드백 통계"""
    defect_type: str
    false_positive: int = Field(default=0)
    false_negative: int = Field(default=0)
    label_correction: int = Field(default=0)

class FeedbackStatsResponse(BaseModel):
    """피드백 통계 응답"""
    total_feedback: int
    by_type: FeedbackTypeStats
    by_defect_type: List[DefectTypeFeedbackStats]
    recent_feedback_count: int
    period_description: str = "최근 24시간"
```

### 2. `database/db.py` (DB 함수 추가)

```python
async def get_feedback_stats() -> Dict:
    """
    피드백 통계 집계

    - 전체 피드백 개수
    - 피드백 종류별 집계 (FP/FN/LC)
    - 결함 타입별 × 피드백 종류별 교차 집계
    - 최근 24시간 피드백 개수
    """
```

**주요 로직:**

1. **전체 피드백 개수**: `SELECT COUNT(*) FROM feedback`
2. **피드백 종류별 집계**: `GROUP BY feedback_type`
3. **결함 타입별 교차 집계**:
   - `inspection_logs`와 JOIN
   - `detections` JSON 파싱
   - 결함 타입별로 피드백 종류 집계
   - 합계 내림차순 정렬
4. **최근 24시간 피드백**: `WHERE created_at >= datetime('now', '-1 day')`

### 3. `routers/feedback.py` (엔드포인트 추가)

```python
@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_statistics():
    """
    피드백 통계 조회

    MLOps 모니터링용:
    - 피드백 종류별 집계 (FP/FN/LC)
    - 결함 타입별 피드백 집계
    - 최근 24시간 피드백 개수
    """
```

## API 사용법

### 요청

```bash
GET /feedback/stats
```

### 응답 (200 OK)

```json
{
  "total_feedback": 8,
  "by_type": {
    "false_positive": 2,
    "false_negative": 2,
    "label_correction": 4
  },
  "by_defect_type": [
    {
      "defect_type": "scratch",
      "false_positive": 2,
      "false_negative": 0,
      "label_correction": 4
    }
  ],
  "recent_feedback_count": 8,
  "period_description": "최근 24시간"
}
```

## 응답 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `total_feedback` | int | 전체 피드백 개수 |
| `by_type` | object | 피드백 종류별 집계 |
| `by_type.false_positive` | int | 오탐 개수 (정상인데 불량으로) |
| `by_type.false_negative` | int | 미탐 개수 (불량인데 정상으로) |
| `by_type.label_correction` | int | 라벨 수정 개수 |
| `by_defect_type` | array | 결함 타입별 피드백 집계 |
| `by_defect_type[].defect_type` | string | 결함 종류 |
| `by_defect_type[].false_positive` | int | 해당 결함의 오탐 개수 |
| `by_defect_type[].false_negative` | int | 해당 결함의 미탐 개수 |
| `by_defect_type[].label_correction` | int | 해당 결함의 라벨 수정 개수 |
| `recent_feedback_count` | int | 최근 24시간 피드백 개수 |
| `period_description` | string | 집계 기간 설명 |

## 테스트 방법

### 1. 자동 테스트

```bash
cd serving/api

# 서버 시작 (터미널 1)
uv run uvicorn main:app --reload --port 8000

# 테스트 실행 (터미널 2)
uv run python test/test_feedback_stats.py
```

**예상 결과:**
```
============================================================
  피드백 통계 API 테스트
============================================================
✅ 서버 상태: OK (Status 200)
✅ PASS GET /feedback/stats

📊 통계 요약:
   총 피드백: 8개
   오탐 (FP): 2개
   미탐 (FN): 2개
   라벨 수정: 4개
   최근 24시간: 8개
   집계 기간: 최근 24시간

   결함 타입별 집계:
   - scratch: FP=2, FN=0, LC=4 (합계: 6)

🎉 모든 테스트 통과!
```

### 2. 수동 테스트 (curl)

```bash
curl http://localhost:8000/feedback/stats | jq '.'
```

### 3. Swagger UI

http://localhost:8000/docs#/Feedback/get_feedback_statistics_feedback_stats_get

## 데이터베이스 확인

```bash
# 피드백 종류별 집계
sqlite3 data/inspection.db "
SELECT feedback_type, COUNT(*) as count
FROM feedback
GROUP BY feedback_type;
"

# 최근 24시간 피드백
sqlite3 data/inspection.db "
SELECT COUNT(*) as recent_count
FROM feedback
WHERE created_at >= datetime('now', '-1 day');
"

# 결함 타입별 피드백 (샘플)
sqlite3 data/inspection.db "
SELECT il.detections, f.feedback_type
FROM feedback f
INNER JOIN inspection_logs il ON f.log_id = il.id
WHERE il.result = 'defect'
LIMIT 3;
"
```

## 성능 고려사항

1. **인덱스 활용**
   - `idx_feedback_type`: 피드백 종류별 집계 최적화
   - `idx_feedback_created_at`: 시간순 필터링 최적화
   - `idx_feedback_log_id`: JOIN 성능 최적화

2. **JSON 파싱 에러 처리**
   - `detections` JSON 파싱 실패 시 무시 (try-except)
   - 잘못된 데이터로 인한 API 실패 방지

3. **정렬**
   - 결함 타입별 통계는 합계 내림차순 정렬
   - 가장 많은 피드백을 받은 결함 타입이 먼저 표시

## MLOps 활용 예시

### 1. 모델 개선 우선순위 결정

```json
{
  "by_defect_type": [
    {
      "defect_type": "scratch",
      "false_positive": 50,  // 오탐이 많음
      "false_negative": 20,  // 미탐도 많음
      "label_correction": 10
    },
    {
      "defect_type": "hole",
      "false_positive": 5,
      "false_negative": 30,  // 미탐이 특히 많음
      "label_correction": 2
    }
  ]
}
```

**우선순위:**
- `scratch`: 오탐이 많음 → 정밀도 개선 필요
- `hole`: 미탐이 많음 → 재현율 개선 필요

### 2. 재학습 데이터 준비

```bash
# 피드백이 있는 이미지 목록 조회 (향후 구현)
GET /feedback/queue

# 결과: 재라벨링이 필요한 이미지 목록
```

### 3. 모델 성능 추적

```json
{
  "total_feedback": 150,
  "recent_feedback_count": 12,  // 최근 24시간
  "by_type": {
    "false_positive": 80,
    "false_negative": 45,
    "label_correction": 25
  }
}
```

**추세 분석:**
- 피드백이 계속 증가 → 모델 성능 저하 의심
- 최근 피드백 급증 → 생산 환경 변화 확인

## 구현 통계

- **추가 코드**: ~140줄
- **수정 파일**: 3개
- **테스트 코드**: ~200줄
- **구현 시간**: ~30분

## 관련 문서

- **상세 계획**: `reactive-popping-hejlsberg.md`
- **Feedback API 문서**: `FEEDBACK_API.md`
- **테스트 가이드**: `test/TEST_GUIDE.md`
