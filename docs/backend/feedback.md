# 피드백 시스템

품질 관리자가 AI 검사 결과를 검토하고 피드백을 제공하는 MLOps 시스템입니다.

## 피드백 수집 (POST /feedback/bulk)

1개 PCB(log_id)에 대해 여러 bbox 피드백을 한 번에 제출하는 Bulk 방식입니다.

### 피드백 종류

| 타입 | 설명 | target_bbox | correct_label |
|------|------|-------------|---------------|
| false_positive | 정상인데 불량으로 오탐 | 필수 | 불필요 |
| tp_wrong_class | 결함은 맞지만 종류가 틀림 | 필수 | 필수 |
| false_negative | 불량인데 정상으로 미탐 | 불필요 | 불필요 |

`correct_label` 허용값: scratch, hole, contamination, crack, normal

### 처리 플로우

1. 원본 detections 조회
2. 각 bbox에 대해 피드백 매칭 (target_bbox 좌표 비교)
3. 최종 라벨 생성:
   - 피드백 없음 → 원본 유지 (암묵적 TP)
   - `false_positive` → 라벨에서 삭제
   - `tp_wrong_class` → 클래스 수정
4. S3 refined/ 폴더에 이미지 복사 + YOLO 라벨 저장
5. DB에 피드백 저장 + `is_verified = true` 설정
6. FALSE_NEGATIVE가 있으면 needs_labeling/ 폴더에 복사

## 자동 재라벨링

피드백이 제출되면 별도의 승인 과정 없이 즉시 학습 데이터가 생성됩니다.

### 최종 라벨 생성 로직

```
원본 detection     +  피드백             =  최종 라벨
────────────────────────────────────────────────────
[scratch, bbox_A]  +  (피드백 없음)      →  원본 유지 (암묵적 TP)
[hole, bbox_B]     +  false_positive     →  라벨에서 삭제
[hole, bbox_C]     +  tp_wrong_class     →  클래스를 scratch로 수정
(없음)             +  false_negative     →  needs_labeling/ 폴더로 분류
```

### YOLO 포맷 변환

픽셀 좌표 `[x1, y1, x2, y2]`를 YOLO 정규화 좌표 `[x_center, y_center, width, height]`로 변환합니다.

```python
def normalize_bbox(bbox_px, width, height):
    x1, y1, x2, y2 = bbox_px
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return [x_center, y_center, w, h]
```

결함 타입은 `CLASS_ID_MAP`에 따라 정수 ID로 변환됩니다: scratch=0, hole=1, contamination=2, crack=3.

### S3 저장 구조

```
s3://pcb-data-storage/
├── raw/                          # 원본 검사 이미지
│   └── 20260118/
│       └── 2026-01-18T15-01-00_PCB_002.jpg
├── refined/                      # 검증 완료 학습 데이터
│   ├── images/
│   │   └── 42_2026-01-18T15-01-00_PCB_002.jpg
│   └── labels/
│       └── 42_2026-01-18T15-01-00_PCB_002.txt
└── needs_labeling/               # False Negative (추가 라벨링 필요)
    ├── images/
    │   └── 42_2026-01-18T15-01-00_PCB_002.jpg
    └── metadata/
        └── 42_2026-01-18T15-01-00_PCB_002.json
```

- `refined/`: 피드백이 반영된 최종 학습 데이터. YOLO 포맷 라벨 파일 포함.
- `needs_labeling/`: False Negative 항목. 메타데이터 JSON 포함.

## 검증 상태 추적

피드백이 처리되면 해당 `inspection_logs` 행에 검증 완료를 표시합니다:

```sql
UPDATE inspection_logs
SET is_verified = 1,
    verified_at = '2026-01-30T10:30:45',
    verified_by = 'qa_team'
WHERE id = 42
```

## 피드백 통계 (GET /feedback/stats)

모델 성능을 정량적으로 분석하기 위한 통계 API입니다.

### image_stats (이미지 레벨 검증 진행률)

전체 이미지 중 검증 완료된 비율을 추적합니다.

### bbox_stats (bbox 레벨 정확도 분석)

검증 완료된 불량 이미지의 각 bbox에 대해, 피드백이 없으면 정답(암묵적 True Positive), 피드백이 있으면 해당 타입으로 분류합니다.

**결함 타입별 세부 정확도:**
- scratch: 정확도 90.0%
- hole: 정확도 77.1%

### class_confusion (클래스 혼동 행렬)

`tp_wrong_class` 피드백으로부터 모델의 체계적인 오분류 패턴을 추출합니다.

**예시:**
- `from_class: "hole"`, `to_class: "scratch"`, `count: 3` → 모델이 scratch를 hole로 3번 잘못 예측

## MLOps 활용

### 모델 개선 우선순위 결정

`bbox_stats.by_defect_type`에서 결함 타입별 정확도를 비교하여, 정확도가 낮은 타입을 우선적으로 개선 대상으로 선정합니다.

### 재학습 데이터셋 자동 구성

`GET /feedback/export` 엔드포인트는 학습 서버에서 호출하여 S3 데이터셋 위치와 이미지 수를 확인할 수 있습니다.

**예시:**

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

학습 서버에서 `aws s3 sync` 명령으로 데이터셋을 로컬에 동기화하면, 바로 YOLO 모델 재학습에 투입할 수 있습니다.

### 재학습 시점 판단

다음 지표들의 변화 추이를 모니터링하여 재학습 시점을 판단합니다:

- 헬스 모니터링의 평균 신뢰도가 지속적으로 하락하는 경우
- 피드백 통계의 `accuracy_rate`가 목표 수준 이하로 떨어지는 경우
- 특정 결함 타입의 false_positive 또는 tp_wrong_class가 급증하는 경우
- `class_confusion`에서 특정 클래스 쌍의 혼동이 반복되는 경우

## 관련 파일

- `routers/feedback.py`: 피드백 API 라우터
- `utils/s3_dataset.py`: S3 데이터셋 관리, YOLO 포맷 변환
- `database/db.py`: 피드백 CRUD, 통계 함수
- `schemas/schemas.py`: 피드백 관련 Pydantic 모델
