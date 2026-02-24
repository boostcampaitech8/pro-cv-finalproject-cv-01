# MLOps 자동 배포 파이프라인 — 변경 사항 및 가이드

> **작성일**: 2026-02-24  
> **대상**: 팀원 전체  
> **목적**: git pull 이후 변경된 코드, MLOps 사이클 전체 흐름, 사용 방법, 추후 작업 정리

---

## 1. 변경 파일 목록

### 수정된 파일 (4개)

| 파일 | 경로 | 변경 요약 |
|---|---|---|
| `register_model.py` | `training/scripts/` | `latest.json`에 `tags` 필드 추가 |
| `config.py` | `serving/edge/` | TRT 빌드 상수, 폴링 주기, 조건 필터, Golden Set 경로 추가 |
| `updater.py` | `serving/edge/` | 전면 재작성 — 조건 필터, atomic symlink, evaluate_engine |
| `inference_worker.py` | `serving/edge/` | `threading.Lock` 추가로 Hot-Swap Race Condition 제거 |

### 신규 생성 파일 (2개)

| 파일 | 경로 | 용도 |
|---|---|---|
| `test_update_cycle.sh` | `serving/edge/` | 업데이트 사이클 단위 테스트 (16개 케이스) |
| `test_filter_logic.py` | `serving/edge/` | 태그 조건 필터 검증 (4개 케이스) |

---

## 2. 파일별 상세 변경 내용

### `training/scripts/register_model.py`

**변경 이유**: Jetson이 모델을 선별 수락하려면 `latest.json`에 태그 정보가 필요합니다.

**변경 내용**: `--tags` 인자에서 파싱한 태그를 `latest.json`에 딕셔너리로 포함합니다.

```python
# 변경 전
latest_info = {
    "version": version.version,
    "s3_key": s3_key,
    "run_id": run_id,
    "timestamp": int(time.time() * 1000)
}

# 변경 후 — tags 필드 추가됨
latest_info = {
    "version": version.version,
    "s3_key": s3_key,
    "run_id": run_id,
    "tags": {"status": "retrained"},   # ← 신규
    "timestamp": int(time.time() * 1000)
}
```

---

### `serving/edge/config.py`

**변경 이유**: TRT 빌드 파라미터가 하드코딩되어 있었고, Jetson Orin Nano Super(8GB 통합 메모리) 사양에 맞는 안전한 값이 필요했습니다.

**추가된 상수들**:

| 상수 | 기본값 | 환경변수 | 설명 |
|---|---|---|---|
| `TRT_WORKSPACE_MB` | 2048 | `TRT_WORKSPACE_MB` | TRT 빌드 시 최대 워크스페이스 (MB) |
| `TRT_MIN_TIMING_ITERS` | 1 | `TRT_MIN_TIMING_ITERS` | 최소 타이밍 반복 (빌드 속도 우선) |
| `TRT_AVG_TIMING_ITERS` | 8 | `TRT_AVG_TIMING_ITERS` | 평균 타이밍 반복 (NVIDIA 권장) |
| `TRT_BUILD_TIMEOUT_S` | 600 | `TRT_BUILD_TIMEOUT_S` | 빌드 최대 허용 시간 (초) |
| `MODEL_POLL_INTERVAL` | 300 | `MODEL_POLL_INTERVAL` | S3 폴링 주기 (초, 기본 5분) |
| `REQUIRED_TAG_KEY` | `"status"` | `REQUIRED_TAG_KEY` | 모델 수락 조건 태그 키 |
| `REQUIRED_TAG_VALUE` | `"retrained"` | `REQUIRED_TAG_VALUE` | 모델 수락 조건 태그 값 |
| `GOLDEN_SET_DIR` | `serving/edge/golden_set/` | — | 고정 검증 데이터셋 경로 |
| `GOLDEN_YAML_PATH` | `golden_set/golden.yaml` | — | YOLO val() 설정 파일 경로 |

> 모든 환경변수는 선택 사항이며, 설정하지 않으면 위 기본값이 적용됩니다.  
> Jetson마다 환경변수로 개별 설정할 수 있습니다.

---

### `serving/edge/updater.py` (전면 재작성)

**변경 이유**: 기존 코드의 빈틈 5가지를 보완했습니다.

| # | 기존 문제 | 해결 |
|---|---|---|
| 1 | 조건 필터 없음 → 모든 모델 무조건 수락 | `_meets_condition()`으로 태그 확인 후 불충족 시 스킵 |
| 2 | `--workspace=1024` 하드코딩 | `config.TRT_WORKSPACE_MB` 등 상수 참조 |
| 3 | trtexec 무한 대기 가능 | `timeout=600s` 설정 |
| 4 | 심링크 삭제→생성 사이 깨진 경로 | `os.symlink(tmp)` → `os.rename()` atomic 교체 |
| 5 | `evaluate_engine()` mock random | Golden Set 기반 실제 `YOLO.val()` 구현 |

**중점으로 볼 부분**:
- `_meets_condition()` (L58~L74): 태그 필터 로직
- `download_and_build()` (L100~L163): BUILDING_FLAG + trtexec + timeout
- `switch_model()` (L190~L210): atomic symlink 교체
- `evaluate_engine()` (L172~L216): Golden Set 없으면 0.0 반환 (경고 출력)

---

### `serving/edge/inference_worker.py`

**변경 이유**: `reload_model()`이 `self.model`을 교체하는 순간, 다른 스레드에서 `predict()`가 동시에 같은 `self.model`을 사용할 수 있어 크래시 위험(Race Condition)이 있었습니다.

**핵심 변경**:
```python
self._model_lock = threading.Lock()  # __init__에 추가

# reload_model(): Lock 안에서 포인터 교체
with self._model_lock:
    old_model = self.model
    self.model = new_model

# run(): Lock 안에서 predict 실행
with self._model_lock:
    results = self.model.predict(crop, conf=0.25, verbose=False)
```

> **추론은 TRT 빌드 중에도 멈추지 않습니다.**  
> `BUILDING_FLAG`는 `inference_worker`가 읽지 않습니다 (기존 설계와 동일).  
> Lock은 `reload_model()` 실행 시에만 짧게(~수십ms) 대기합니다.

---

## 3. 사용 방법

### 테스트 실행

```bash
cd serving/edge
bash test_update_cycle.sh
```

16개 테스트가 실행되며, 모두 `✅ PASS`가 출력되면 정상입니다.

### Jetson에서 실행

```bash
# 환경변수 설정 (선택)
export MODEL_POLL_INTERVAL=60        # 1분마다 폴링 (테스트용)
export REQUIRED_TAG_KEY=status
export REQUIRED_TAG_VALUE=retrained

# updater 단독 실행
python serving/edge/updater.py

# 또는 main.py에서 전체 시스템 실행 (updater 포함)
python serving/edge/main.py
```

---

## 4. MLOps 사이클 전체 흐름

### 데이터 흐름도

```
[1] Jetson — RTSP 수신 & 추론
    serving/edge/main.py (오케스트레이터)
    serving/edge/inference_worker.py (추론 수행)
        ↓ 결과: {image_id, camera_id, defects[], image(base64)}

[2] Jetson → API 서버 전송
    serving/edge/upload_worker.py → POST /detect/
        ↓ 결과: 백엔드로 추론 결과 + 이미지 전달

[3] API 서버 — DB + S3 저장
    serving/api/routers/detect.py
      → SQLite: inspection_logs 테이블에 기록
      → S3: raw/ 폴더에 원본 이미지 저장
      → Slack: 상태 변화 시 알림 (선택)
        ↓ 결과: DB에 검사 로그, S3에 원본 이미지

[4] 대시보드 — 피드백 대기열 조회
    serving/api/routers/feedback.py → GET /feedback/queue
      → 검증되지 않은 inspection_logs 목록 반환
        ↓ 결과: 사용자가 대시보드에서 오탐/미탐/클래스오류 확인

[5] 사용자 — 피드백 제출
    serving/api/routers/feedback.py → POST /feedback/bulk
      → FP(오탐) 제거, 클래스 수정, FN(미탐) 추가
        ↓ 결과: 검수 완료된 라벨

[6] API → S3 refined 저장
    serving/api/utils/s3_dataset.py
      → S3 refined/ 폴더에 이미지 + YOLO 라벨(.txt) 저장
      → FN이 있으면 needs_labeling/ 으로 이미지 이동  ← (수동 라벨링 필요)
        ↓ 결과: S3에 학습 가능한 정제 데이터

[7] Airflow — 주간 재학습 파이프라인
    training/dags/pcb_retrain.py (DAG 오케스트레이터)
    ├─ training/scripts/sync_data.py      → S3 refined/ 데이터 병합
    ├─ training/scripts/run_exp.py        → FP32 모델 학습
    ├─ training/scripts/train_qat.py      → QAT(양자화 인식) 학습
    ├─ training/scripts/export_qat.py     → ONNX 변환 (메타데이터 주입)
    └─ training/scripts/register_model.py → MLflow 등록 + S3 업로드
        ↓ 결과: S3 models/candidates/{version}_best.onnx + latest.json

[8] Jetson — 자동 폴링 & 모델 교체  ← (★ 이번에 구현한 부분)
    serving/edge/updater.py
    ├─ S3 latest.json 폴링 (기본 5분)
    ├─ 태그 조건 필터 (status=retrained)
    ├─ ONNX 다운로드 + trtexec TRT 변환
    ├─ Golden Set 평가 (mAP50 비교)
    └─ atomic symlink 교체 → RELOAD_FLAG
         ↓
    serving/edge/inference_worker.py
      → RELOAD_FLAG 감지 → Lock 획득 → 모델 교체
         ↓ [1]로 복귀 — 새 모델로 추론 시작
```

### 각 단계의 입력 → 출력 연결

| 단계 | 입력 | 처리 주체 | 출력 | 다음 단계 입력 |
|---|---|---|---|---|
| ① 추론 | RTSP 영상 | `inference_worker.py` | 결함 bbox + 이미지 | ② |
| ② 전송 | 추론 결과 JSON | `upload_worker.py` | HTTP 요청 | ③ |
| ③ 저장 | HTTP 요청 | `detect.py` | SQLite row + S3 raw/ | ④ |
| ④ 피드백 대기 | SQLite 미검증 로그 | `feedback.py` (GET) | 대기열 JSON | ⑤ |
| ⑤ 피드백 제출 | 사용자 입력 | `feedback.py` (POST) | 라벨 수정/삭제/추가 | ⑥ |
| ⑥ 정제 저장 | 검수된 라벨 | `s3_dataset.py` | S3 refined/ | ⑦ |
| ⑦ 재학습 | S3 refined/ 병합 | Airflow DAG | ONNX + latest.json | ⑧ |
| ⑧ 배포 | latest.json + ONNX | `updater.py` | TRT engine + RELOAD_FLAG | ① |

### 유기성 진단

현재 **①~⑧ 전체 사이클이 코드로 연결**되어 있습니다.  
유일한 **수동 개입 지점**은 다음 두 곳입니다:

1. **④~⑤ 피드백 제출**: 사용자가 대시보드에서 오탐/미탐을 검수합니다.
2. **⑥ FN(미탐) 후처리**: FN 피드백이 있으면 `needs_labeling/`으로 이미지가 이동되며, 수동 라벨링 후 `refined/`에 추가해야 다음 학습에 포함됩니다.

---

## 5. 추후 작업 (TODO)

### 🔴 필수 — 실제 운영 전 반드시 완료

| 작업 | 이유 |
|---|---|
| **Golden Dataset 준비** | `evaluate_engine()`이 현재 golden.yaml 없이 `mAP=0.0`을 반환하므로, 신규 모델이 **성능 검증 없이 항상 채택**됩니다. ~100장의 고정 검증 이미지+라벨을 Jetson의 `serving/edge/golden_set/`에 준비해야 합니다. |
| **Jetson 실기기 E2E 테스트** | 현재 단위 테스트(16개)는 로컬에서 S3 없이 실행됩니다. 실제 S3 연동, trtexec 빌드, 핫스왑까지의 전체 사이클을 Jetson에서 검증해야 합니다. |

### Golden Dataset 마련 방법

```
serving/edge/golden_set/
  images/           ← PCB 검증 이미지 ~100장
  labels/           ← YOLO 형식 라벨 (.txt)
  golden.yaml       ← YOLO 데이터셋 설정

# golden.yaml 예시
path: /home/jetson/PCB/serving/edge/golden_set
train: images
val:   images
nc: 6
names: {0: missing_hole, 1: mouse_bite, 2: open_circuit,
        3: short, 4: spur, 5: spurious_copper}
```

**소스**: `training/PCB_DATASET/val.txt`에서 약 100장을 샘플링하여 고정합니다.  
재학습 때마다 바뀌지 않는 **고정 기준 세트**여야 모델 간 비교가 의미 있습니다.

### 🟡 권장

| 작업 | 이유 |
|---|---|
| **needs_labeling → refined 자동화** | 현재 FN 피드백 이미지는 수동으로 라벨링해야 합니다. CVAT 등의 라벨링 도구와 연동하면 이 과정을 반자동화할 수 있습니다. |
| **모델 배포 Slack 알림** | 현재 Slack 알림은 `detect.py`의 시스템 상태 변화에만 적용됩니다. `updater.py`의 `report_status()`에 Slack 알림을 추가하면 모델 배포 성공/실패를 팀에 자동 공유할 수 있습니다. |
| **`/feedback/export` API 활용** | `feedback.py`에 이미 `GET /feedback/export` 엔드포인트가 구현되어 있습니다. Airflow의 `sync_data.py`에서 이 API를 호출하여 새 데이터 유무를 확인하고, 데이터가 없으면 학습을 건너뛰는 로직을 추가할 수 있습니다. |

### 🟢 선택

| 작업 | 이유 |
|---|---|
| **모델 롤백 기능** | 새 모델 배포 후 문제가 발생할 때 이전 엔진으로 되돌리는 기능. 현재는 `models/` 디렉토리에 이전 버전 엔진이 남아 있으므로, 심링크만 되돌리면 됩니다. |
| **다중 Jetson 관리** | 여러 대의 Jetson이 있을 때 `EDGE_DEVICE_ID` 환경변수로 구분하고, MLflow에 디바이스별 배포 이력을 기록합니다. 현재도 지원되지만 운영 시 모니터링 대시보드가 필요합니다. |

---

## 6. 디렉토리 구조 요약

```
PCB/
├── training/                        ← 학습 서버 (Vast)
│   ├── dags/
│   │   └── pcb_retrain.py            # Airflow DAG: 주간 재학습 오케스트레이션
│   └── scripts/
│       ├── sync_data.py              # S3 refined/ → 로컬 병합
│       ├── run_exp.py                # FP32 학습
│       ├── train_qat.py              # QAT 학습
│       ├── export_qat.py             # ONNX 변환
│       └── register_model.py         # ★ MLflow 등록 + S3 업로드 + latest.json (tags 포함)
│
├── serving/
│   ├── api/                         ← 백엔드 API 서버 (EC2)
│   │   ├── routers/
│   │   │   ├── detect.py             # Jetson → DB + S3 raw/ 저장
│   │   │   ├── feedback.py           # 피드백 → S3 refined/ 저장
│   │   │   └── sessions.py           # 세션 관리
│   │   ├── database/db.py            # SQLite 조작
│   │   └── utils/
│   │       ├── s3_dataset.py         # S3 refined/ 저장 로직
│   │       └── slack_notifier.py     # Slack 알림
│   │
│   └── edge/                        ← Jetson 디바이스
│       ├── config.py                 # ★ TRT 상수 + 폴링/필터 환경변수 + Golden Set 경로
│       ├── main.py                   # 전체 오케스트레이터
│       ├── inference_worker.py       # ★ 추론 + threading.Lock 핫스왑
│       ├── updater.py                # ★ S3 폴링 → TRT 빌드 → 모델 교체
│       ├── upload_worker.py          # 결과를 API로 전송
│       ├── test_update_cycle.sh      # ★ 단위 테스트 (16개)
│       └── test_filter_logic.py      # ★ 태그 필터 테스트 (4개)
```

> ★ 표시가 이번에 수정/생성된 파일입니다.
