# Main / Updater 변경 요약 및 운영 가이드

목적

- `main.py`가 RTSP/추론 중단으로 인해 전체 프로세스를 멈추지 않도록 하고,
- `updater.py`(모델 다운로드 → ONNX→TensorRT 빌드 → 프로모션)가 추론 중일 때는 안전하게 동작을 보류하도록 하기 위함입니다.

핵심 목표

1. `main`은 멈추지 않는다 — RTSP 연결이 끊겨도 주기적 재접속을 계속 시도합니다.
2. `updater`는 실제 추론(이미지 전처리→예측)이 활발할 때 빌드/프로모션을 하지 않습니다.
3. `updater`가 빌드 중일 때 추론이 활성화되면, `updater`는 빌드 프로세스를 중단하고 임시 파일을 정리합니다.

수정된 파일 (요약)

- [work/pro-cv-finalproject-cv-01/serving/edge/config.py](work/pro-cv-finalproject-cv-01/serving/edge/config.py)
  - `INFERENCE_BUSY_FLAG_PATH` 추가: models 디렉터리의 `.inference_busy` 파일 경로
  - `RTSP_RETRY_INTERVAL_S` 추가 (기본 300초)
  - `INFERENCE_IDLE_TIMEOUT_S` 추가: 추론 비활성으로 판단하기 위한 idle 타임아웃

- [work/pro-cv-finalproject-cv-01/serving/edge/rtsp_receiver.py](work/pro-cv-finalproject-cv-01/serving/edge/rtsp_receiver.py)
  - RTSP 연결이 끊겼을 때 스레드를 완전히 종료하지 않고 `RTSP_RETRY_INTERVAL_S` 간격으로 재접속을 시도하도록 변경
  - 잘못된 상대 import를 절대 경로로 수정
  - `is_running()`가 실제 연결 상태를 반영하도록 개선

- [work/pro-cv-finalproject-cv-01/serving/edge/inference_worker.py](work/pro-cv-finalproject-cv-01/serving/edge/inference_worker.py)
  - `models/.inference_busy` 플래그의 의미를 개선: 워커가 단순히 실행 중인 상태가 아닌, 실제로 최근에 예측(크롭 처리)을 한 경우에 플래그를 생성/갱신하고, 일정 시간(`INFERENCE_IDLE_TIMEOUT_S`)의 무활동 후 삭제하도록 변경
  - 이로써 RTSP가 끊겨 있거나 추론이 진짜로 유휴 상태일 때 `updater`가 빌드를 진행할 수 있게 됨

- [work/pro-cv-finalproject-cv-01/serving/edge/updater.py](work/pro-cv-finalproject-cv-01/serving/edge/updater.py)
  - 전체 설계 변경 요약:
    - 업데이트 루프가 `models/.inference_busy`를 체크하여 추론이 활성화되어 있으면 폴링/다운로드/빌드를 보류
    - 빌드 시작 전 `_wait_until_idle(context)`로 문맥을 포함해 주기적으로(로그와 함께) 대기
    - `download_and_build()`는 `self.current_build_files`에 현재 빌드 대상 파일들을 기록하여 중단 시 안전 삭제
    - `trtexec` 실행은 `subprocess.Popen`으로 수행하되, 프로세스 그룹(preexec/setpgid)을 만들어 중단 시 `os.killpg()`로 그룹 전체를 종료하도록 개선 (쉘-wrapper가 남아 `pgrep`에 두 줄 보이는 상황 완화)
    - `_kill_trtexec_if_running()` 추가: 빌드 중 추론 활성화가 감지되면 트텐서 프로세스 그룹을 강제 종료하고 `BUILDING_FLAG` 및 임시 파일을 삭제
    - CLI 확장: `--status`(플래그/프로세스 현황), `--abort`(빌드 중단 및 정리)
    - `updater.run()`의 들여쓰기/루프 버그 수정

- [work/pro-cv-finalproject-cv-01/serving/edge/main.py](work/pro-cv-finalproject-cv-01/serving/edge/main.py)
  - `main`에서 `updater.py`를 자동으로 실행하지 않도록 변경 (업데이터는 별도 서비스/유닛으로 운영)
  - `--num-cameras`(`-n`) 처리 로직 보강: 입력이 RTSP base URL인 경우 `_1/_2/...` suffix를 붙여 다중 카메라 URL을 생성
  - 모든 수신기가 오프라인이거나 프레임 큐가 비어 있어도 프로세스가 종료되지 않고 재접속을 계속 시도하도록 변경

동작 원리 (요약)

- 추론 바쁨 플래그: `models/.inference_busy`
  - 이전: 워커 시작/종료로 플래그를 관리
  - 현재: 실제 예측이 발생할 때 플래그를 생성하거나 최종 접근 시간을 갱신하고, `INFERENCE_IDLE_TIMEOUT_S` 초 동안 예측이 없으면 플래그를 삭제
  - 이 변경으로 `updater`는 모델이 실질적으로 유휴일 때만 빌드를 시도합니다.

- updater의 빌드/중단 동작
  - 빌드 전: `_wait_until_idle("build start")`로 플래그가 사라질 때까지 대기
  - 빌드 중: `trtexec`는 별도 프로세스 그룹으로 실행되며, 주기적으로 `models/.inference_busy`를 확인
    - 추론 활성화가 감지되면 `updater`는 프로세스 그룹을 종료(`os.killpg()`), `self.current_build_files`에 기록된 임시 파일을 삭제하고 `BUILDING_FLAG`를 제거
  - `trtexec`가 쉘을 통해 실행되면 `pgrep`에 두 줄(쉘 + trtexec)이 보일 수 있음 — 이제 프로세스 그룹을 사용해 안전 종료합니다.

운영/사용 가이드

- main 실행 (업데이터는 별도 실행):

```bash
python work/pro-cv-finalproject-cv-01/serving/edge/main.py -i <RTSP_BASE_URL> -n <num_cameras>
```

예: `-i rtsp://host:8554/pcb_stream -n 2` → `pcb_stream_1`, `pcb_stream_2` 생성

- updater 수동/서비스로 실행:

```bash
# 폴링/업데이트 루프
python work/pro-cv-finalproject-cv-01/serving/edge/updater.py

# 현재 상태 확인 (플래그 / trtexec 프로세스)
python updater.py --status

# 강제 중단 및 정리 (trtexec kill, v*.onnx/.engine 삭제, 플래그 삭제)
python updater.py --abort
```

로깅/확인 포인트

- RTSP 재접속 로그 (rtsp_receiver):
  - 연결 시: `[RTSPReceiver] 연결 성공: WxH @ fps` 
  - 끊김 시: `[RTSPReceiver] 연결 끊김 - ${RTSP_RETRY_INTERVAL_S}s 후 재시도`

- 추론 바쁨 플래그 (inference_worker):
  - `models/.inference_busy`는 예측 활동이 있을 때만 존재하며, `INFERENCE_IDLE_TIMEOUT_S` 이후 삭제됩니다.
  - `updater.py` 로그에서 `🔒 추론 중이므로 전체 업데이트 사이클 대기...` 또는 `🔒 추론 바쁨 (download start) — 대기 중...` 메시지를 확인합니다.

- trtexec/빌드 중단 로그 (updater):
  - `⛔ trtexec PGID 12345 강제 종료 (업데이트 일시 중단)` 
  - `  빌드 플래그도 함께 삭제됨` 
  - `  빌드 중이던 파일 삭제: /.../vX.onnx` 등

검증 체크리스트

1. main 시작 → `models/.inference_busy` 파일이 (예측 활동 시) 생성/갱신되는지 확인
2. 다른 터미널에서 `python updater.py` 실행 → 추론 활동이 있으면 `updater`가 대기 로그를 출력하는지 확인
3. main이 완전히 유휴(또는 종료) 상태일 때 `updater`가 폴링 → 다운로드 → 빌드를 재개하는지 확인
4. `updater`가 빌드 중일 때 main에서 예측 활동이 발생하면 `trtexec` 프로세스 그룹이 종료되고 관련 파일이 삭제되는지 확인
5. RTSP 연결 끊김 시 `RTSPReceiver`가 재접속 시도 로그를 남기며 프로세스를 유지하는지 확인

디버깅 팁

- 현재 `trtexec` 프로세스/그룹 확인:

```bash
pgrep -af trtexec || echo 'no process'
ps -o pid,ppid,pgid,cmd -p <PID>
```

- `trtexec`가 두 줄로 보일 때(쉘 + trtexec)는 `updater`가 `shell=True`로 실행했기 때문일 수 있습니다. 안전하게 하려면 `shell=False`로 직접 실행하거나, 프로세스 그룹을 만들고 `os.killpg()`로 종료하도록 개선했습니다.

- 현재 빌드/플래그 상태 확인:

```bash
python updater.py --status
ls -la work/pro-cv-finalproject-cv-01/serving/edge/models/
```

주의사항 및 트레이드오프

- 워커가 실제 예측을 수행하는 동안에는 `updater`가 빌드를 시도하지 않으므로 신모델 채택이 지연될 수 있습니다. (의도된 동작 — 추론 우선)
- `trtexec`를 강제 종료하면 빌드가 불완전하게 끝나 리소스가 일시적으로 많이 소비될 수 있습니다. 문제가 계속되면 빌드 타임아웃/리트라이 설정을 조정하거나 빌드 호스트를 분리하는 것을 권장합니다.

향후 개선 제안

- local HTTP control endpoint를 만들어 `updater`가 `main`에 교체 요청을 보내고 `main`이 이를 수락/거부하면 더 안전합니다.
- `INFERENCE_BUSY_FLAG` 대신 IPC(Unix socket)나 간단한 RPC를 사용하면 파일 race 및 잔존 파일 문제를 줄일 수 있습니다.


생성일: 2026-02-26

**실시간 로그 예시 (case 별)**

아래는 운영 중에 실제로 출력되는 로그 라인들과, 각 로그가 어떤 상황을 의미하는지 정리한 예시입니다.

- RTSP 연결
  - 연결 시작: `[RTSPReceiver] 시작 (재접속 허용): {source}`
  - 연결 시도(디버그): `[RTSPReceiver] 연결 시도: {source}`
  - 연결 성공: `[RTSPReceiver] 연결 성공: 1920x1080 @ 30.0fps`
  - 소스 열기 실패(재시도): `[RTSPReceiver] 소스를 열 수 없습니다: {source} — 300s 후 재시도`
  - 스트림 일시 중단(읽기 실패): `[RTSPReceiver] 스트림 일시 중단: 연속 {N}회 읽기 실패 — 재접속 시도`
  - 연결 끊김(재시도 예정): `[RTSPReceiver] 연결 끊김 - 300s 후 재시도`

- 추론 워커(busy 플래그)
  - busy 생성(처음 예측 시작 시): `추론 활성화 — busy 플래그 생성`
  - 추론 시간 로깅: `[InferenceWorker][cam_1] 추론 시간: 12.3ms`
  - busy 제거(유휴): `추론 비활성화 — busy 플래그 제거 (idle)`
  - 워커 중지 시: (종료 루틴에서 `.inference_busy` 파일 삭제)

- Updater 대기/폴링 관련
  - 폴링 시작: `[YYYY-MM-DD HH:MM:SS] S3 업데이트 확인 중...`
  - 이미 최신: `  최신 버전 유지 중 (vX)`
  - 새 버전 발견: `🌟 새 버전 발견: vX (현재: vY)`
  - 조건 불충분으로 스킵: `⏭  조건 불충족 → 스킵: status=None (필요: 'retrained')`
  - 추론 바쁨으로 대기(다운로드 전): `🔒 추론 바쁨 (download start) — 대기 중...` 또는 `🔒 추론 바쁨 — 대기 중...`
  - 주기적 대기 로그: `🔒 추론이 계속 바쁨 — 10초 후 재확인`
  - 전체 사이클 대기: `🔒 추론 중이므로 전체 업데이트 사이클 대기...`

- ONNX 다운로드 / 빌드
  - 다운로드 시작: `📥 S3에서 다운로드 중: {s3_key} → /.../vX.onnx`
  - 다운로드 완료: `✅ 다운로드 완료: /.../vX.onnx`
  - 빌드 준비: `🔨 TensorRT 엔진 빌드 준비 (workspace=1024MB, avgTiming=8)...`
  - 빌드 성공: `✅ TRT 엔진 빌드 성공!`
  - 빌드 실패: `❌ TRT 빌드 실패: <error>`
  - 빌드 타임아웃: `❌ TRT 빌드 타임아웃 (600초 초과)`

- trtexec 프로세스 감지/중단
  - `pgrep`에 두 줄 보이는 경우: `/bin/sh -c trtexec ...` (쉘 래퍼) 와 `trtexec --onnx=...` (실제 프로세스)
    - 원인: `subprocess.Popen(cmd, shell=True)`로 실행 시 쉘 프로세스가 먼저 보임
    - 개선: updater는 프로세스 그룹을 사용(preexec/setpgid)하거나 `shell=False`로 직접 실행하도록 변경했습니다.
  - 강제 종료 로그: `⛔ trtexec PID 12345 강제 종료 (업데이트 일시 중단)`
  - 플래그/파일 정리: `  빌드 플래그도 함께 삭제됨` / `  빌드 중이던 파일 삭제: /.../vX.onnx`

- 평가 및 프로모션
  - Golden Set 평가 시작: `   🔍 Golden Set 평가 중: /.../vX.engine`
  - 평가 완료: `   ✅ 평가 완료 → mAP50=0.9123`
  - 채택 메시지: `🎉 신규 모델 채택!`
  - 미달 메시지: `📉 신규 모델 성능 미달 → 기각`

- MLflow (선택적)
  - MLflow 미존재 경고: `Warning: mlflow package not found. Continuing without MLflow tracking.`
  - MLflow URI 로깅: `📡 MLflow Tracking URI: http://...`
  - 스테이지 전환 시작: `🚀 MLflow Registry: vX → Production 승격 중...`
  - 스테이지 전환 완료: `✅ MLflow 업데이트 완료: vX → Production`

- 사용자 명령 (`--status`, `--abort`)
  - `--status` 예시 출력:
```
BUILDING_FLAG_PATH: /.../models/.building_engine
  내용: building v12
현재 실행 중인 trtexec 프로세스:
   118283 /bin/sh -c trtexec --onnx=... --saveEngine=... --int8 --fp16 ...
   118284 trtexec --onnx=... --saveEngine=... --int8 --fp16 ...
```
  - `--abort` 예시 출력:
```
빌드 플래그 발견: building v12
  trtexec PID 118284 종료
  삭제: /.../v12.onnx
  빌드 플래그 삭제
빌드 중단 및 정리 완료
```

이 섹션은 실제 운영 중 출력되는 로그를 근거로 했으며, 추후 로그 포맷이나 수준을 중앙 로거로 통일하면 더 쉽게 파싱/모니터링할 수 있습니다.

