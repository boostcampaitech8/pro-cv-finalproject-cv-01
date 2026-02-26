# 🚀 GPU 서버 초기 세팅 가이드

> RunPod 등 외부 GPU 서버를 새로 빌릴 때, 이 문서만 따라하면 재학습 환경이 완성됩니다.  
> **최종 수정**: 2026-02-25

## 📋 목차

1. [사전 조건 확인](#1-사전-조건-확인)
2. [Git 설정 및 프로젝트 클론](#2-git-설정-및-프로젝트-클론)
3. [uv 패키지 매니저 설치](#3-uv-패키지-매니저-설치)
4. [Python 가상환경 + 의존성 설치](#4-python-가상환경--의존성-설치)
5. [AWS CLI 설치 & S3 연결](#5-aws-cli-설치--s3-연결)
6. [MLflow 서버 실행](#6-mlflow-서버-실행)
7. [Airflow 실행](#7-airflow-실행)
8. [SSH 서버 설정 (팀원 초대)](#8-ssh-서버-설정-팀원-초대)
9. [팀원 root SSH 접속 설정 (공개키 등록)](#9-팀원-root-ssh-접속-설정-공개키-등록)
10. [웹 UI 접속 방법 (RunPod Public URL)](#10-웹-ui-접속-방법-runpod-public-url)
11. [전체 검증](#11-전체-검증)
12. [트러블슈팅 모음](#12-트러블슈팅-모음)

---

## 1. 사전 조건 확인

서버 접속 후 GPU 및 기본 환경을 확인합니다.

```bash
nvidia-smi          # GPU 확인 (CUDA 버전 포함)
python3 --version   # 시스템 Python 버전 확인
cat /etc/os-release # OS 확인
```

| 항목 | 권장 사양 |
|------|----------|
| GPU | CUDA 12.x 이상 (RTX 3090/4090/A100 등) |
| OS | Ubuntu 20.04+ |
| Python | 3.10.x (정확한 버전은 uv가 자동 설치) |

### RunPod Pod 생성 시 주의사항

Pod을 새로 만들 때 **Expose HTTP Ports** 항목에 `5000, 8082`를 추가하면,
브라우저에서 MLflow/Airflow에 직접 접속할 수 있습니다.
이 설정은 **Pod 생성 시에만** 가능하며, 이미 생성된 Pod은 SSH 터널링으로 접속해야 합니다.
(자세한 내용은 [10번](#10-로컬-pc에서-웹-ui-접속-ssh-터널링) 참고)

---

## 2. Git 설정 및 프로젝트 클론

```bash
git config --global user.name "본인이름"
git config --global user.email "본인이메일@example.com"

cd /workspace
git clone <REPO_URL>
```

---

## 3. uv 패키지 매니저 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 후 **PATH 적용** (중요!):
```bash
source $HOME/.local/bin/env
# 영구 적용
echo 'source $HOME/.local/bin/env' >> ~/.bashrc
```

> ⚠️ `uv: command not found` 에러 시 `source $HOME/.local/bin/env` 실행

---

## 4. Python 가상환경 + 의존성 설치

```bash
# 프로젝트가 요구하는 정확한 Python 버전 설치
uv python install 3.10.19

# 가상환경 생성 + 의존성 동기화
cd /workspace/pro-cv-finalproject-cv-01/training
uv venv
uv sync
```

### ⚠️ Python 버전 불일치 에러

```
error: The Python request from `.python-version` resolved to Python 3.10.12,
which is incompatible with the project's Python requirement: `==3.10.19`
```

**해결**:
```bash
uv python install 3.10.19
rm -rf .venv           # 기존 venv 삭제
uv venv                # 3.10.19로 재생성
uv sync                # 의존성 재설치
```

### ⚠️ 중요: uv가 Python을 `/root/.local/` 아래에 설치함

이 경로는 기본적으로 root만 접근 가능합니다.
팀원(merby 등)이 같은 서버를 쓸 경우, [9번 항목](#9-팀원-계정-생성--권한-설정)에서 권한 설정이 필요합니다.

---

## 5. AWS CLI 설치 & S3 연결

### 5-1. AWS CLI v2 설치

```bash
apt-get update -qq && apt-get install -y -qq unzip

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
unzip -o /tmp/awscliv2.zip -d /tmp/
/tmp/aws/install
```

### 5-2. AWS 자격증명 설정

```bash
aws configure
```

| 프롬프트 | 입력값 |
|---------|--------|
| AWS Access Key ID | jetson .env 확인 |
| AWS Secret Access Key | jetson .env 확인 |
| Default region name | `ap-southeast-2` |
| Default output format | `json` |

### 5-3. S3 연결 테스트

```bash
aws s3 ls s3://pcb-data-storage/
# 정상 출력: PRE PCB_DATASET/ PRE models/ PRE raw/ PRE refined/
```

---

## 6. MLflow 서버 실행

```bash
VENV=/workspace/pro-cv-finalproject-cv-01/training/.venv/bin

nohup $VENV/mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri file:///workspace/pro-cv-finalproject-cv-01/training/mlruns \
  > /tmp/mlflow.log 2>&1 &

echo "MLflow started on port 5000"
```

검증:
```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:5000/
# 출력: HTTP 200
```

### ⚠️ MLflow가 안 뜸

**원인**: `source .venv/bin/activate` 방식은 `nohup`에서 환경이 유지 안 됨  
**해결**: venv의 `mlflow` 바이너리를 **절대 경로로 직접 호출**

```bash
# ❌ 안 되는 방식
source .venv/bin/activate && nohup mlflow server ...

# ✅ 되는 방식
nohup /workspace/.../training/.venv/bin/mlflow server ...
```

---

## 7. Airflow 실행

### ⚠️ 핵심: 포트 8082를 사용해야 함

RunPod 컨테이너에는 **nginx가 8081을 이미 점유**하고 있습니다.
따라서 Airflow는 **8082** 포트를 사용합니다.

### 7-1. start_airflow.sh 사용

```bash
cd /workspace/pro-cv-finalproject-cv-01/training
bash start_airflow.sh
```

> 💡 이제 `start_airflow.sh`는 자동으로 **백그라운드(`nohup`)**에서 실행됩니다. 터미널을 닫아도 프로세스가 유지됩니다.
> `sh start_airflow.sh`로 실행하면 `source: not found` 에러 발생. 반드시 **`bash`**로 실행!

### 7-2. 수동 실행 (start_airflow.sh 없이)

```bash
export PATH="/workspace/pro-cv-finalproject-cv-01/training/.venv/bin:$HOME/.local/bin:$PATH"
export AIRFLOW_HOME="/workspace/pro-cv-finalproject-cv-01/training"
export AIRFLOW_WEBSERVER_PORT=8082
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8082
export AIRFLOW__API__BASE_URL="http://localhost:8082"
export AIRFLOW__CORE__EXECUTION_API_SERVER_URL="http://localhost:8082/execution/"
export AIRFLOW__CORE__EXECUTOR="LocalExecutor"

nohup airflow standalone > /tmp/airflow.log 2>&1 &
```

### 7-3. airflow.cfg 경로 확인 (중요!)

Airflow 최초 실행 시 `airflow.cfg`가 자동 생성되는데,
**`dags_folder` 등의 경로가 실제 프로젝트 경로와 다를 수 있습니다.**

```bash
grep -E "dags_folder|plugins_folder|sql_alchemy|base_log_folder" \
  /workspace/pro-cv-finalproject-cv-01/training/airflow.cfg
```

모든 경로가 `/workspace/pro-cv-finalproject-cv-01/training/...`으로 되어 있어야 합니다.
만약 `/workspace/final_project/training/...` 등 다른 경로가 보이면:

```bash
sed -i 's|/workspace/final_project/training|/workspace/pro-cv-finalproject-cv-01/training|g' \
  /workspace/pro-cv-finalproject-cv-01/training/airflow.cfg
```

수정 후 Airflow 재시작 필요.

### 7-4. Airflow 로그인 정보

최초 실행 시 admin 패스워드가 자동 생성됩니다:
```bash
cat /workspace/pro-cv-finalproject-cv-01/training/simple_auth_manager_passwords.json.generated
```

| 항목 | 값 |
|------|---|
| 사용자명 | `admin` |
| 비밀번호 | 위 파일에서 확인 |

### 7-5. DAG에 pcb_retrain_pipeline이 안 보일 때

1. `airflow.cfg`의 `dags_folder` 경로 확인 (7-3 참고)
2. DAG 파일(`dags/pcb_retrain.py`)의 Python import 에러 확인:
   ```bash
   /workspace/pro-cv-finalproject-cv-01/training/.venv/bin/python \
     /workspace/pro-cv-finalproject-cv-01/training/dags/pcb_retrain.py
   ```

### 7-6. Airflow에서 S3 동기화가 `NoCredentialsError`로 실패할 때

Airflow의 BashOperator 서브프로세스가 AWS 자격증명을 못 찾는 문제입니다.
`dags/pcb_retrain.py`에 환경변수를 설정해야 합니다:

```python
# dags/pcb_retrain.py 상단에 추가
AWS_ENV = {
    'HOME': '/root',
    'AWS_SHARED_CREDENTIALS_FILE': '/root/.aws/credentials',
    'AWS_CONFIG_FILE': '/root/.aws/config',
}

# S3를 사용하는 BashOperator에 env= 추가
t1_sync = BashOperator(
    task_id='sync_data_from_s3',
    bash_command=f'{PYTHON_BIN} {SCRIPTS_DIR}/sync_data.py',
    cwd=PROJECT_ROOT,
    env=AWS_ENV        # ← 이 줄 추가
)
```

---

## 8. SSH 서버 설정 (팀원 초대)

RunPod 컨테이너는 `systemd`가 없으므로 `systemctl` 대신 `service` 명령어를 사용합니다.

### 8-1. SSH 설치 및 시작

```bash
apt-get update -qq && apt-get install -y -qq openssh-server

# systemctl은 안 됨! service 사용
mkdir -p /run/sshd
service ssh start
```

### 8-2. SSH 보안 설정

```bash
# 공개키 인증만 허용 (권장)
sed -i 's/^#\?PubkeyAuthentication .*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# 필요하면 root 로그인 허용 (팀원이 root로 접속해야 할 때)
sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config

# 비밀번호 인증 끄기 (공개키만 사용하려면)
# sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config

# 설정 적용
service ssh restart
```

---

## 9. 팀원 root SSH 접속 설정 (공개키 등록)

팀원들이 별도의 계정 생성 없이, 본인의 PC에서 서버의 **root** 계정으로 바로 접속할 수 있게 설정합니다.

### 9-1. root 공개키 등록 (가장 중요)

팀원들에게 공개키(`ssh-ed25519 AAAA...` 또는 `ssh-rsa AAAA...`)를 받아서 아래 파일에 추가합니다.

- **수정할 파일**: `/root/.ssh/authorized_keys`

```bash
# 디렉토리 생성 및 권한 설정
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# 공개키 추가 (기존 키가 있다면 뒤에 추가됨)
cat >> /root/.ssh/authorized_keys << 'EOF'
ssh-ed25519 AAAA... user1@pc
ssh-ed25519 AAAA... user2@pc
EOF

# 권한 설정 (본인만 읽을 수 있어야 함)
chmod 600 /root/.ssh/authorized_keys
```

### 9-2. root 로그인 허용 설정

```bash
# root 로그인 설정을 yes로 변경
sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config

# 설정 적용을 위해 SSH 재시작
service ssh restart
```

### 9-3. 접속 확인 (팀원 PC에서)

팀원은 로그라이크(RunPod)에서 제공하는 SSH 명령어를 그대로 사용하되, 별도의 비밀번호 없이 바로 접속됩니다.
```bash
ssh -p <포트> root@<IP>
```

---

---

## 10. 웹 UI 접속 방법 (RunPod Public URL)

### 방법 1: RunPod에서 포트 직접 열기 (권장)

RunPod Pod 생성 시 **Expose HTTP Ports** 항목에 `5000, 8082`를 추가했다면, 별도의 터널링 없이 바로 접속 가능합니다.

1. RunPod 대시보드에서 해당 Pod의 **Connect** 버튼 클릭
2. **HTTP Forwarding** 또는 **Exposed Ports** 탭 확인
3. 생성된 URL 클릭:
   - `https://xxx-5000.proxy.runpod.net` → MLflow
   - `https://xxx-8082.proxy.runpod.net` → Airflow

### 방법 2: SSH 터널링 (포트가 안 열려있을 때만 사용)

포트 설정을 깜빡했다면 로컬 터미널에서 아래 명령을 실행한 뒤 브라우저에서 `localhost:18082` 등으로 접속합니다.

```bash
ssh -p <포트> -L 18082:localhost:8082 -L 15000:localhost:5000 root@<IP>
```

---

## 11. 전체 검증

모든 세팅이 완료되면 아래 체크리스트로 확인합니다:

```bash
# 1. Python 가상환경
/workspace/pro-cv-finalproject-cv-01/training/.venv/bin/python --version
# → Python 3.10.19

# 2. S3 연결
aws s3 ls s3://pcb-data-storage/refined/
# → refined/ 폴더 내 이미지/라벨 목록

# 3. MLflow
curl -s -o /dev/null -w "MLflow: HTTP %{http_code}\n" http://localhost:5000/
# → HTTP 200

# 4. Airflow
curl -s -o /dev/null -w "Airflow: HTTP %{http_code}\n" http://localhost:8082/
# → HTTP 200

# 5. GPU
/workspace/pro-cv-finalproject-cv-01/training/.venv/bin/python -c \
  "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# → CUDA: True, GPU: NVIDIA GeForce RTX 4090

# 6. pcb_retrain_pipeline DAG 로딩 확인
grep "pcb_retrain" /tmp/airflow.log | tail -1
# → pcb_retrain.py 로딩 로그

# 7. SSH 서비스
ps -ef | grep sshd | grep -v grep | head -1
# → sshd 프로세스 정보
```

---

## 12. 트러블슈팅 모음

이 프로젝트에서 실제로 겪은 에러와 해결법입니다.

### 환경 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| `uv: command not found` | PATH 미적용 | `source $HOME/.local/bin/env` |
| Python 3.10.12 vs 3.10.19 | 시스템 Python과 프로젝트 요구 버전 불일치 | `uv python install 3.10.19` → `uv venv` → `uv sync` |
| `bash: unzip: command not found` | 경량 Docker 이미지 | `apt-get install -y unzip` |
| `bash: nano: command not found` | 경량 Docker 이미지 | `apt-get install -y nano` 또는 `cat` / `sed` 사용 |

### Airflow 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| `source: not found` | `sh`로 실행 | **`bash`** `start_airflow.sh`로 실행 |
| `FileNotFoundError: 'airflow'` | venv/bin이 PATH에 없음 | `export PATH=".venv/bin:$PATH"` 후 실행 |
| 8081 포트에서 Airflow 대신 다른 페이지 | RunPod nginx가 8081 점유 | **8082** 포트 사용 |
| DAG에 pcb 안 보임 | `airflow.cfg`의 `dags_folder` 경로 오류 | `sed`로 경로 수정 후 Airflow 재시작 |
| `NoCredentialsError` | Airflow subprocess에서 AWS 자격증명 못 찾음 | DAG에 `env=AWS_ENV` 추가 (7-6 참고) |

### SSH / 팀원 접속 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| `systemctl: command not found` | 컨테이너에 systemd 없음 | `service ssh start` 사용 |
| `Permission denied (publickey)` | `.ssh` 폴더 소유자가 root | `chown -R 유저:유저 /home/유저/.ssh` |
| `Permission denied (publickey)` | `PermitRootLogin no` | `sed -i 's/^PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config` → `service ssh restart` |
| 팀원이 Python/MLflow 실행 안 됨 | `/root/.local/` 접근 권한 없음 | `chmod -R a+rX /root/.local/` |
| 팀원이 AWS S3 접근 안 됨 | `~/.aws` 없음 | `ln -sf /root/.aws /home/유저/.aws` |
| `bind: Permission denied` (로컬) | 로컬 포트 권한 부족 | 포트 번호를 18082, 15000 등으로 변경 |
| 브라우저에서 `Exposed Ports` 페이지 | RunPod 프록시 URL 사용 | `http://localhost:포트` 사용 (SSH 터널 필요) |

### S3 관련

| 증상 | 원인 | 해결 |
|------|------|------|
| `Unable to locate credentials` | `aws configure` 미실행 | `aws configure` 실행 |
| S3 접근은 되는데 Airflow에서 실패 | subprocess 환경 문제 | DAG에 `env=AWS_ENV` 추가 |

---

## 📁 포트 요약

| 서비스 | 내부 포트 | 외부 접근 방법 |
|--------|----------|---------------|
| MLflow | 5000 | SSH 터널: `-L 15000:localhost:5000` → `http://localhost:15000` |
| Airflow | 8082 | SSH 터널: `-L 18082:localhost:8082` → `http://localhost:18082` |
| SSH | 22 | RunPod 외부 포트 (대시보드에서 확인) |

> ⚠️ **8081은 사용하지 마세요!** RunPod의 nginx가 점유하고 있습니다.

---

## 🔄 재학습 파이프라인 흐름

```
[Airflow DAG: pcb_retrain_pipeline]

1. sync_data_from_s3      → S3 refined/ 에서 재학습 데이터 동기화
2. train_fp32_model       → FP32 모델 학습 (config.yaml 기반)
3. train_qat_model        → QAT 양자화 학습 (config_qat.yaml 기반)
4. export_onnx            → ONNX 모델 추출
5. register_model         → MLflow 등록 + S3 업로드 (→ 엣지 배포)
```

> 매주 일요일 UTC 18:00 (한국 시간 월요일 03:00) 자동 실행 예약됨

---

## 💡 Quick Start (서버를 새로 팠을 때)

```bash
# ==============================================================
# 1단계: 기본 도구 설치
# ==============================================================
apt-get update -qq && apt-get install -y -qq unzip openssh-server sudo nano

# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

# ==============================================================
# 2단계: Python + 의존성
# ==============================================================
cd /workspace/pro-cv-finalproject-cv-01/training
uv python install 3.10.19 && uv venv && uv sync

# ==============================================================
# 3단계: AWS CLI
# ==============================================================
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -o /tmp/awscliv2.zip -d /tmp/ && /tmp/aws/install
aws configure
# → Key: (팀 노션 참고)
# → Secret: (팀 노션 참고)
# → Region: ap-southeast-2
# → Format: json

# ==============================================================
# 4단계: MLflow 시작
# ==============================================================
nohup .venv/bin/mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri file://$(pwd)/mlruns > /tmp/mlflow.log 2>&1 &

# ==============================================================
# 5단계: Airflow 시작 (포트 8082!)
# ==============================================================
bash start_airflow.sh
# → 이제 백그라운드에서 실행되므로, 같은 터미널에서 바로 다음 작업 가능

# ==============================================================
# 6단계: SSH + 팀원 root 접속 설정
# ==============================================================
mkdir -p /run/sshd && service ssh start
sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh restart

# 팀원 공개키 등록 (수정할 파일: /root/.ssh/authorized_keys)
mkdir -p /root/.ssh && chmod 700 /root/.ssh
cat >> /root/.ssh/authorized_keys << 'EOF'
팀원_공개키_1
팀원_공개키_2
EOF
chmod 600 /root/.ssh/authorized_keys

# ==============================================================
# 7단계: 검증
# ==============================================================
curl -s -o /dev/null -w "MLflow: %{http_code}\n" http://localhost:5000/
curl -s -o /dev/null -w "Airflow: %{http_code}\n" http://localhost:8082/
aws s3 ls s3://pcb-data-storage/ | head -3
ps -ef | grep sshd | head -1
echo "✅ 세팅 완료!"
```
