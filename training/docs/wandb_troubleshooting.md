# WandB 연동 트러블슈팅 기록

> 작성일: 2026-02-23  
> 환경: RunPod (NVIDIA RTX 4090, Ubuntu, Python 3.11)  
> 목표: `scripts/run_exp.py` 실행 → WandB에 학습 로그 및 모델 아티팩트 업로드 → Jetson에서 다운로드

---

## 새 환경 셋업 체크리스트

> **다른 서버에서 처음 세팅할 때 이 순서대로 진행하면 모든 에러를 회피할 수 있습니다.**

```bash
# ① 의존성 설치 (반드시 training/ 안에서)
cd /workspace/pro-cv-finalproject-cv-01/training
/root/.local/bin/uv sync

# ② 가중치 파일 정상 여부 확인 (110MB 이상이어야 정상)
ls -lh pretrained_weights/yolo11x.pt

# 없거나 손상됐다면 재다운로드
mkdir -p pretrained_weights
./.venv/bin/python -c "
from ultralytics import YOLO; import shutil
YOLO('yolo11x.pt')
shutil.move('yolo11x.pt', 'pretrained_weights/yolo11x.pt')
print('Done')
"

# ③ 디스크 용량 확인 (중복 .venv 주의)
df -h .
du -sh ../.venv 2>/dev/null && echo "⚠️ 중복 venv 존재! 삭제 필요" || echo "OK"

# ④ WandB 로그인 (처음 한 번만 / 또는 환경변수로 대체)
wandb login
# 또는: export WANDB_API_KEY=wandb_v1_XXXX...

# ⑤ .netrc 키 중복 여부 확인
python -c "
import re
txt = open('/root/.netrc').read()
key = re.search(r'password (.+)', txt).group(1)
print('Key duplicated!' if key.count('wandb') > 1 else 'Key OK')
"

# ⑥ 학습 실행
./.venv/bin/python -u scripts/run_exp.py --config configs/config.yaml
```

---

## 최종 실행 명령어

```bash
# training/ 디렉토리 안에서 실행해야 함
cd /workspace/pro-cv-finalproject-cv-01/training
./.venv/bin/python -u scripts/run_exp.py --config configs/config.yaml
```

---

## 발생한 에러 목록 및 해결 과정

---

### 에러 1: `No such file or directory`

**에러 메시지**
```
/venv/bin/python3: can't open file '/workspace/pro-cv-finalproject-cv-01/scripts/run_exp.py': 
[Errno 2] No such file or directory
```

**원인**  
CWD(현재 작업 디렉토리)가 프로젝트 루트(`/workspace/pro-cv-finalproject-cv-01`)인 상태에서 `scripts/run_exp.py`를 상대 경로로 참조했기 때문.  
실제 스크립트는 `training/scripts/run_exp.py`에 위치한다.

**해결**  
항상 `training/` 폴더 안으로 이동한 뒤 실행한다.

```bash
cd /workspace/pro-cv-finalproject-cv-01/training
./.venv/bin/python -u scripts/run_exp.py --config configs/config.yaml
```

**고려할 점**  
- `uv run`은 `--project` 플래그로 프로젝트 루트를 지정해도, Python 인터프리터의 CWD는 쉘의 CWD를 따른다.
- `scripts/run_exp.py` 내부에서 `os.path.dirname(__file__)` 기반으로 경로를 계산하므로, 스크립트 위치 기준의 상대 경로 탐색은 정상 동작한다.

---

### 에러 2: `Disk quota exceeded`

**에러 메시지**
```
OSError: [Errno 122] Disk quota exceeded
```

**원인**  
볼륨(20GB)이 꽉 찬 상태. 원인은 두 개의 가상환경이 중복 생성되었기 때문.

| 경로 | 크기 |
|------|------|
| `/workspace/pro-cv-finalproject-cv-01/.venv` | **8.6 GB** |
| `/workspace/pro-cv-finalproject-cv-01/training/.venv` | **10 GB** |

`uv pip install wandb`를 루트 경로에서 실행하여 바깥쪽에 불필요한 가상환경이 생성되었다.

**해결**  
```bash
# 중복 가상환경 삭제
rm -rf /workspace/pro-cv-finalproject-cv-01/.venv

# uv 캐시도 정리 (최대 14GB+ 절약)
/root/.local/bin/uv cache clean
```

**고려할 점**  
- `uv` 명령 실행 시 항상 어느 경로에서 실행하는지 확인할 것.
- `training/` 폴더에만 `.venv`가 있어야 한다.
- `df -h`의 용량과 실제 RunPod 대시보드의 Volume 용량이 다를 수 있으므로, 동시에 두 가지를 확인할 것.

---

### 에러 3: `PytorchStreamReader failed reading zip archive`

**에러 메시지**
```
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
```

**원인**  
디스크가 꽉 찬 상태에서 `yolo11x.pt` 가중치 파일을 다운로드하다 중단되어 파일이 손상(corrupt)됨.  
정상 파일 크기: **~114 MB** → 손상된 파일 크기: **64 MB**

```bash
# 손상 확인
ls -lh pretrained_weights/yolo11x.pt
# 110MB 미만이면 비정상 → 삭제 후 재다운로드
```

**해결**  
```bash
rm -f yolo11x.pt pretrained_weights/yolo11x.pt
mkdir -p pretrained_weights
./.venv/bin/python -c "
from ultralytics import YOLO; import shutil
YOLO('yolo11x.pt')
shutil.move('yolo11x.pt', 'pretrained_weights/yolo11x.pt')
print('Done:', __import__('os').path.getsize('pretrained_weights/yolo11x.pt') // 1e6, 'MB')
"
```

**고려할 점**  
- 디스크 용량 부족 상황에서 다운로드가 중단되어도 오류 메시지 없이 파일만 불완전하게 저장될 수 있다.
- `src/models/yolov11x.py`에서 가중치 탐색 순서: `./yolo11x.pt` → `pretrained_weights/yolo11x.pt` → Ultralytics 자동 다운로드

---

### 에러 4: `Invalid project name: cannot contain '/'`

**에러 메시지**
```
wandb.errors.UsageError: Invalid project name 'ckgqf1313-boostcamp/PODO': 
cannot contain characters '/,\,#,?,%,:', found '/'
```

**원인**  
`config.yaml`의 `wandb_project`가 `entity/project` 형식인데, 기존 코드에서 이를 `project=` 인자 하나로 통째로 넘겼기 때문.

**기존 코드 (버그)**
```python
# src/train.py
wandb.init(project=self.config['wandb_project'], ...)
# → project='ckgqf1313-boostcamp/PODO' (슬래시 포함 → 에러)
```

**수정된 코드** (`src/train.py` — 이미 반영됨)
```python
wandb_project_str = self.config['wandb_project']
if '/' in wandb_project_str:
    wandb_entity, wandb_project = wandb_project_str.split('/', 1)
else:
    wandb_entity, wandb_project = None, wandb_project_str

wandb.init(
    entity=wandb_entity,    # 'ckgqf1313-boostcamp'
    project=wandb_project,  # 'PODO'
    name=f"{self.config['exp_name']}",
    reinit=True
)
```

**고려할 점**  
- `config.yaml`의 `wandb_project`는 `entity/project` 형식으로 유지하되, 코드에서 파싱해야 한다.
- `entity`를 `None`으로 넘기면 기본 개인 계정으로 로깅된다.

---

### 에러 5: `401 Unauthorized - user is not logged in`

**에러 메시지**
```
wandb.errors.CommError: Error uploading run: returned error 401: 
{"errors":[{"message":"user is not logged in"}]}
```

**원인**  
`wandb login` 실행 중 `Ctrl+C`로 중단되어 `/root/.netrc`에 API 키가 **두 번 중복**으로 저장됨.

```
# 손상된 .netrc (이 상태면 401 에러)
password wandb_v1_QD2phLZ...wandb_v1_QD2phLZ...
```

**해결**  

방법 A: `.netrc` 직접 수정
```
machine api.wandb.ai
  login user
  password wandb_v1_XXXXXXXXXXXXXXXXXXXXXXXX
```

방법 B: `--relogin` 으로 덮어쓰기
```bash
wandb login --relogin
```

방법 C: 환경변수 사용 (`.netrc` 없이도 동작)
```bash
export WANDB_API_KEY=wandb_v1_XXXX...
```

**키 중복 여부 빠른 확인:**
```bash
python -c "
import re
txt = open('/root/.netrc').read()
key = re.search(r'password (.+)', txt).group(1)
print('⚠️ 중복!' if key.count('wandb') > 1 else '✅ 정상')
"
```

---

### 에러 6: `NameError: name 'os' is not defined` (아티팩트 미업로드)

**에러 메시지**
```
NameError: name 'os' is not defined
  File "src/train.py", line 248, in train
    best_model_path = os.path.join(self.model.trainer.save_dir, "weights", "best.pt")
```

**원인**  
`src/train.py` 파일 상단에 `import os`가 누락되어 있었다.  
학습 자체는 정상 완료되지만, 완료 후 아티팩트 업로드 함수(`upload_to_wandb`) 호출 전에 에러가 발생하여 **WandB 아티팩트에 모델이 올라가지 않는다.**

**수정된 코드** (`src/train.py` — 이미 반영됨)
```python
# 파일 최상단에 추가
import os

class PCBTrainer:
    ...
```

**고려할 점**  
- WandB **Run** 탭에는 학습 로그가 정상 기록되더라도, **Artifacts** 탭이 비어있다면 이 에러일 가능성이 높다.
- 학습 완료 로그 마지막에 `[WandB] uploading ...` 메시지가 없으면 업로드 실패로 판단한다.

---

### 에러 7: `WandB Artifact 다운로드 후 파일 삭제 시 재다운로드 안 됨`

**증상**
```
[OK] 이미 최신 버전입니다. 새 다운로드 불필요.
[FAIL] 모델을 가져오지 못했습니다.
```

**원인**  
버전 파일(`.model_version` / `.test_model_version`)만 남아있고 실제 `.pt` 파일은 삭제된 상태에서,  
기존 코드가 **버전 ID만 비교**하여 "업데이트 없음"으로 판단하고 다운로드를 건너뜀.

**수정된 코드** (`model_manager.py`, `test_artifact_download.py` — 이미 반영됨)  
버전 일치 + 실제 파일 존재 여부를 모두 확인:

```python
# 버전이 같아도 파일이 없으면 재다운로드
if latest_version == self.current_version and self._local_files_exist():
    return None  # 진짜 업데이트 없음

# 아니면 → 재다운로드 진행
```

**고려할 점**  
- Jetson 재부팅 후 모델 파일이 사라진 경우에도 자동 복구된다.
- 버전 파일만 삭제하는 것으로도 강제 재다운로드를 트리거할 수 있다.

---

### 에러 8: `CUDA out of memory` (추론 중)

**에러 메시지**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.77 GiB.
GPU 0 has a total capacity of 23.52 GiB of which 3.73 GiB is free.
```

**원인**  
학습 직후 GPU 메모리가 충분히 해제되지 않은 상태에서 바로 추론(inference)을 실행했기 때문.  
학습 프로세스가 약 19.78 GiB를 점유한 채로 추론이 시작되었다.

**해결**  
학습 완료 후 별도 프로세스로 추론을 실행하거나, 학습 스크립트에 메모리 정리 코드를 추가:

```python
# 학습 완료 후
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

또는 학습과 추론을 분리하여 각각 독립 실행:
```bash
# 학습
./.venv/bin/python -u scripts/run_exp.py --config configs/config.yaml

# 추론은 별도 실행
./.venv/bin/python scripts/run_inference.py
```

---

### 에러 9: `InferenceWorker 모니터링 스레드 즉시 종료`

**증상**  
모델 업데이트 모니터링 스레드가 시작되자마자 종료되어 Jetson에서 WandB 체크가 이루어지지 않는다.

**원인**  
`inference_worker.py`의 `_monitor_model_updates()` 에서 `while self.running:` 을 조건으로 사용했는데,  
모니터링 스레드가 시작될 시점에 `self.running = False`(기본값)이므로 루프가 즉시 종료됨.

```python
# 기존 코드 (버그)
def run(self):
    self.running = True  # run()이 호출된 다음에야 True
    ...

def _monitor_model_updates(self):
    while self.running:  # 스레드 시작 시점엔 False → 즉시 종료
        ...
```

**수정된 코드** (`inference_worker.py` — 이미 반영됨)  
전용 플래그 `_monitor_running` 분리:

```python
def __init__(self, ...):
    self.running = False
    self._monitor_running = True  # 초기화 시점에 True
    ...
    self.monitor_thread = threading.Thread(target=self._monitor_model_updates, daemon=True)
    self.monitor_thread.start()

def _monitor_model_updates(self):
    while self._monitor_running:  # 별도 플래그 사용
        ...

def stop(self):
    self.running = False
    self._monitor_running = False  # 함께 종료
```

---

## 수정된 파일 목록 (전체)

| 파일 | 수정 내용 |
|------|-----------|
| `training/src/train.py` | `import os` 추가 + `wandb.init()` entity/project 분리 |
| `serving/edge/model_manager.py` | 파일 존재 여부 체크 추가 → 재다운로드 로직 수정 |
| `serving/edge/inference_worker.py` | `_monitor_running` 플래그 분리로 모니터링 스레드 안정화 |
| `serving/edge/pyproject.toml` | 누락된 의존성 추가 (`wandb`, `ultralytics`, `requests`, `python-dotenv`) |
| `serving/edge/test_artifact_download.py` | WandB Artifact 다운로드 → 추론 테스트 스크립트 (신규) |
| `/root/.netrc` | 중복 API 키 제거 (커밋 제외, 각 환경에서 직접 설정) |

---

## 이식성 참고사항 (새 환경에서 코드 실행 전 필독)

### 1. WandB API 키 설정
`.netrc`는 커밋되지 않으므로, 새 환경에서 반드시 직접 설정:
```bash
wandb login
# 또는
export WANDB_API_KEY=wandb_v1_XXXX...
```

### 2. Jetson(ARM64) 전용 주의사항
- `model_manager.py`의 `model.export(format='engine', ...)` 은 **Jetson 전용** (x86에서 실행 불가)
- Jetson에서 `ultralytics` 설치 시 ARM64용 wheel이 필요할 수 있음
- `pyproject.toml`의 버전 범위가 호환되는지 Jetson 공식 문서에서 확인 필요

### 3. `serving/edge/config.py` 하드코딩 IP 수정 필요
```python
RTSP_URL = "rtsp://3.36.185.146:8554/pcb_stream"  # ← 실제 서버 IP로 변경
API_URL  = "http://3.35.182.98:8080/detect/"       # ← 실제 서버 IP로 변경
```
운영 환경에서는 `.env` 파일 또는 환경변수로 관리 권장:
```bash
export RTSP_URL=rtsp://...
export API_URL=http://...
```

### 4. `uv` 설치 경로 확인
```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
# 설치 후
source ~/.bashrc
```

### 5. 아티팩트 다운로드 테스트 (환경 검증용)
```bash
cd /workspace/pro-cv-finalproject-cv-01/training
./.venv/bin/python ../serving/edge/test_artifact_download.py
# → "[SUCCESS] WandB Artifact → 모델 다운로드 → 추론 파이프라인 검증 완료!" 출력되면 정상
```

---

## WandB 프로젝트 구조

```
WandB Project: ckgqf1313-boostcamp/PODO
├── Runs      → 각 실험의 학습 로그 (loss, mAP 등)
└── Artifacts → best.pt 모델 파일 (학습 완료 후 자동 업로드)
               tags: ['latest', 'production']
```

**배포 워크플로우:**
```
학습 완료 → WandB Artifact 자동 업로드 (latest + production 태그)
                ↓
Jetson ModelManager가 5분마다 폴링
                ↓
새 버전 또는 파일 미존재 감지 → 다운로드 → TRT 변환 → Hot-swap
```

학습 완료 후 WandB 대시보드에서 확인:  
👉 https://wandb.ai/ckgqf1313-boostcamp/PODO
