# RunPod 학습 서버 — 실행 가이드

> **대상**: 이 RunPod 서버에서 할 수 있는 것만 정리  
> **Jetson(엣지)** 관련 내용은 `docs/mlops_auto_deploy_guide.md`를 참고하세요

---

## 0. 서비스 시작 (서버 재부팅 후)

서버 재시작 시 아래 세 가지를 순서대로 띄워야 합니다.

```bash
# ① MLflow (포트 5000)
cd /workspace/pro-cv-finalproject-cv-01/training
nohup .venv/bin/mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri file:///workspace/pro-cv-finalproject-cv-01/training/mlruns \
  > /tmp/mlflow.log 2>&1 &

# ② Airflow (포트 8082)
bash start_airflow.sh
# → 터미널이 점유되므로, 다른 터미널을 열어서 작업하세요
# → 또는 nohup으로 백그라운드 실행:
#   nohup bash start_airflow.sh > /tmp/airflow.log 2>&1 &

# ③ SSH (팀원 접속용)
service ssh start
```

### 서비스 상태 확인

```bash
curl -s -o /dev/null -w "Airflow: %{http_code}\n" http://localhost:8082/
curl -s -o /dev/null -w "MLflow:  %{http_code}\n" http://localhost:5000/
ps -ef | grep sshd | head -1
```

> 세 줄 모두 `200` 또는 프로세스 정보가 나오면 정상입니다.

---

## 1. 로컬 PC에서 웹 UI 접속하기

RunPod은 SSH(22) 포트만 외부에 노출합니다.  
**SSH 터널링**으로 로컬 PC 브라우저에서 접속해야 합니다.

```bash
# 로컬 PC의 CMD/터미널에서 실행 (서버가 아님!)
ssh -p 14647 -L 18082:localhost:8082 -L 15000:localhost:5000 root@103.196.86.77
```

| 서비스 | 브라우저 주소 | 로그인 |
|--------|-------------|--------|
| **Airflow** | `http://localhost:18082` | admin / YnU4KS2wHZBaSvGu |
| **MLflow** | `http://localhost:15000` | 없음 |

> ⚠️ SSH 연결을 유지한 상태에서만 브라우저 접속이 됩니다.

---

## 2. 파이프라인 검증 — 단계별 실행

전체 파이프라인 흐름 (이 서버에서 담당하는 부분):

```
S3 데이터 동기화 → FP32 학습 → QAT 학습 → ONNX 추출 → MLflow 등록 + S3 업로드
```

### 사전 준비

```bash
cd /workspace/pro-cv-finalproject-cv-01/training
VENV=.venv/bin/python
```

### Step 1. S3 데이터 동기화

S3의 `refined/` 폴더에서 새 데이터를 가져옵니다.

```bash
$VENV scripts/sync_data.py
```

**확인할 것**:
- `PCB_DATASET/data_retrain.yaml` 파일이 생성되었는지
- `PCB_DATASET/train_retrain.txt`, `val_retrain.txt`가 존재하는지
- 출력 로그에 다운로드된 이미지 수 확인

```bash
ls -la PCB_DATASET/data_retrain.yaml
wc -l PCB_DATASET/train_retrain.txt PCB_DATASET/val_retrain.txt
```

### Step 2. FP32 모델 학습

```bash
$VENV scripts/run_exp.py \
  --config configs/config.yaml \
  --data PCB_DATASET/data_retrain.yaml \
  --name test_fp32
```

**확인할 것**:
- GPU 사용률: 별도 터미널에서 `nvidia-smi` 실행
- 학습 완료 후 `runs/test_fp32/weights/best.pt` 존재 여부

```bash
ls -lh runs/test_fp32/weights/best.pt
```

### Step 3. QAT (양자화 인식) 학습

```bash
$VENV scripts/train_qat.py \
  --config configs/config_qat.yaml \
  --data PCB_DATASET/data_retrain.yaml \
  --name test_qat \
  --weights runs/test_fp32/weights/best.pt
```

**확인할 것**:
- `runs/qat/test_qat/weights/best.pt` 생성 여부

```bash
ls -lh runs/qat/test_qat/weights/best.pt
```

### Step 4. ONNX 추출

```bash
$VENV scripts/export_qat.py \
  --weights runs/qat/test_qat/weights/best.pt \
  --base-weights runs/test_fp32/weights/best.pt \
  --output runs/qat/test_qat/weights/best.onnx
```

**확인할 것**:
- ONNX 파일 크기 (보통 ~80MB)

```bash
ls -lh runs/qat/test_qat/weights/best.onnx
```

### Step 5. MLflow 등록 + S3 업로드

```bash
$VENV scripts/register_model.py \
  --model-path runs/qat/test_qat/weights/best.onnx \
  --tags "status=retrained"
```

**확인할 것**:

```bash
# S3에 모델 업로드 확인
aws s3 ls s3://pcb-data-storage/models/candidates/

# latest.json에 tags 포함 확인
aws s3 cp s3://pcb-data-storage/models/candidates/latest.json - | python3 -m json.tool

# MLflow UI에서 실험 기록 확인 (브라우저)
# http://localhost:15000
```

---

## 3. Airflow DAG로 전체 파이프라인 한 번에 실행

위 Step 1~5를 Airflow가 자동으로 순서대로 실행합니다.

### 방법 A: 웹 UI에서 트리거

1. `http://localhost:18082` 접속
2. `pcb_retrain_pipeline` DAG 찾기
3. 토글 ON (Unpause) → ▶️ Trigger DAG 클릭
4. DAG 클릭 → Grid/Graph 뷰에서 각 Task 상태 확인
   - 🟢 초록: 성공
   - 🔴 빨강: 실패 → 해당 Task 클릭 → Log 탭에서 에러 확인

### 방법 B: CLI에서 트리거

```bash
export PATH="/workspace/pro-cv-finalproject-cv-01/training/.venv/bin:$PATH"
export AIRFLOW_HOME="/workspace/pro-cv-finalproject-cv-01/training"

# DAG 목록 확인
airflow dags list | grep pcb

# 트리거
airflow dags trigger pcb_retrain_pipeline

# 실행 상태 확인
airflow dags list-runs -d pcb_retrain_pipeline
```

### 방법 C: 개별 Task만 테스트

특정 Task가 잘 되는지만 확인하고 싶을 때:

```bash
# S3 동기화만 테스트
airflow tasks test pcb_retrain_pipeline sync_data_from_s3 2026-01-01

# FP32 학습만 테스트
airflow tasks test pcb_retrain_pipeline train_fp32_model 2026-01-01
```

---

## 4. MLflow에서 모델 확인

### CLI로 확인

```bash
cd /workspace/pro-cv-finalproject-cv-01/training

# 실험 목록
.venv/bin/mlflow experiments search

# 최근 실행 기록
.venv/bin/mlflow runs search --experiment-id 0 --max-results 5
```

### 웹 UI에서 확인

`http://localhost:15000` 접속 후:

| 확인 항목 | 위치 |
|----------|------|
| 학습 기록 (loss, mAP 등) | Experiments → 실험 선택 → 특정 Run 클릭 |
| 등록된 모델 | Models 탭 → PCB 모델 목록 |
| 모델 아티팩트 (ONNX 파일 등) | Run 상세 → Artifacts 탭 |

---

## 5. S3 데이터 확인

```bash
# 피드백 정제 데이터 (재학습 소스)
aws s3 ls s3://pcb-data-storage/refined/images/
aws s3 ls s3://pcb-data-storage/refined/labels/

# 모델 후보군 (Jetson이 이걸 폴링)
aws s3 ls s3://pcb-data-storage/models/candidates/

# 최신 모델 메타데이터
aws s3 cp s3://pcb-data-storage/models/candidates/latest.json - | python3 -m json.tool
```

---

## 6. GPU 모니터링

```bash
# 실시간 GPU 사용률 (1초마다 갱신)
watch -n 1 nvidia-smi

# 한 번만 확인
nvidia-smi
```

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| Airflow 웹 안 뜸 | 프로세스 죽음 | `bash start_airflow.sh` 재실행 |
| MLflow 웹 안 뜸 | 프로세스 죽음 | 0번 섹션의 MLflow 시작 명령 실행 |
| DAG에 pcb 안 보임 | `airflow.cfg`의 `dags_folder` 경로 오류 | `grep dags_folder training/airflow.cfg` 확인 |
| `Permission denied` | merby 계정 권한 부족 | `chmod -R a+rX /root/.local/` 실행 |
| S3 접근 실패 | AWS 자격증명 만료 | `aws configure` 재설정 |
| GPU 안 잡힘 | CUDA 드라이버 문제 | `nvidia-smi` 먼저 확인 |
| `sh start_airflow.sh` 에러 | `sh`(dash) 사용 | `bash start_airflow.sh`로 실행 |

---

## 8. 요약 — 이 서버에서 하는 일

```
이 서버 (RunPod GPU)의 역할:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① S3에서 피드백 데이터 가져오기     → sync_data.py
② FP32 모델 학습                   → run_exp.py
③ QAT 양자화 학습                  → train_qat.py
④ ONNX 변환                       → export_qat.py
⑤ MLflow 등록 + S3 업로드          → register_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
→ 이후 Jetson이 S3에서 모델을 자동으로 가져감
```

> **Airflow**가 위 ①~⑤를 매주 일요일 UTC 18:00에 자동 실행합니다.  
> 수동 테스트는 이 가이드의 2번, 3번을 따라하면 됩니다.
