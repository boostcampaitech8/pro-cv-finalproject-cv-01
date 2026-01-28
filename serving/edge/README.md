# Edge Module


## 현재 상태

| 구성요소 | 상태 | 비고 |
|----------|------|------|
| RTSP 수신 | ✅ 완료 | TCP 모드, 재시도 로직 |
| 배경 캡처 | ✅ 완료 | 스트립 타일링 방식 |
| PCB 크롭 | ✅ 완료 | 경계 체크, 크기 검증 |
| 추론 워커 | ✅ 완료 | TensorRT (FP16) 최적화, 백엔드 전송 |

## 모듈 개요

PCB 결함 탐지 시스템의 **엣지(Jetson) 전처리 파이프라인**.

- **역할**: RTSP 영상 수신 → PCB 감지/크롭 → 추론 Queue 전달
- **실행 환경**: Jetson (로컬 테스트는 macOS/Linux)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ RTSP 수신   │────→│  전처리      │────→│ 추론 워커   │
│ (Thread)    │     │  (Main)      │     │ (Thread)    │
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │                    │
   frame_queue        crop_queue           결과 전송
```

## 기술 스택

- **Python**: >=3.11
- **OpenCV**: 영상 처리
- **NumPy**: 수치 연산
- **Threading/Queue**: 비동기 파이프라인

## 프로젝트 구조

```
├── main.py               # 메인 루프 (통합 및 실행 순서 관리)
├── preprocessor.py       # 전처리 파이프라인 클래스
├── inference_worker.py   # 추론 및 결과 전송 워커
├── rtsp_receiver.py      # RTSP 수신 스레드
├── capture_background.py # 배경 이미지 캡처 도구
├── background.png        # 생성된 배경 이미지
├── best.pt               # YOLOv8 모델 (Push/Pull 필요)
├── pyproject.toml
├── uv.lock
└── README.md             # 이 파일
```

## 개발 환경 설정

```bash
cd serving/edge
uv sync --active
```

## 실행 방법

```bash
cd serving/edge
uv run python main.py
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | RTSP URL 또는 비디오 파일 경로 | `rtsp://3.36.185.146:8554/pcb_stream` |
| `--loop`, `-l` | 비디오 파일 반복 재생 | False |
| `--debug`, `-d` | 디버그 모드 (크롭 이미지 저장) | False |
| `--debug-dir` | 디버그 크롭 저장 디렉토리 | `debug_crops` |
| `--max-crops` | 최대 크롭 개수 (0=무제한) | 0 |

## 핵심 클래스

### PCBPreprocessor

프레임을 처리하여 PCB를 감지하고 크롭하는 클래스.

**상태 머신:**
- `background`: 배경 상태 (PCB 없음)
- `pcb`: PCB 진입 상태

**히스테리시스 임계값:**
- `THRESH_LOW = 15`: 이 이하면 background로 전환
- `THRESH_HIGH = 25`: 이 이상이면 pcb로 전환

**주요 메서드:**
- `process_frame(frame)`: 프레임 처리 후 크롭된 PCB 반환 (없으면 None)

### InferenceWorker

PCB 결함 탐지 및 결과 전송을 담당하는 스레드.

**특징:**
- **TensorRT 최적화**: `.pt` 모델을 Jetson 전용 `.engine`으로 자동 변환 (FP16 적용)
- **비동기 처리**: `crop_queue`에서 이미지를 가져와 메인 루프와 별개로 추론 수행
- **Base64 전송**: 탐지 결과와 이미지를 Base64로 인코딩하여 백엔드 API 서버로 전송
- **리소스 관리**: 엔진 빌드 시 높은 점유율을 고려하여 RTSP 수신보다 먼저 초기화됨

### RTSPReceiver

RTSP 스트림을 수신하는 데몬 스레드.

**특징:**
- TCP 모드 사용 (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`)
- 연속 실패 30회까지 재시도 (RTSP 연결 안정성)
- Queue가 가득 차면 오래된 프레임 버림 (실시간성 우선)
- graceful shutdown 지원

## 전처리 로직

### 1. PCB 진입 감지

```python
# 검사 영역 (세로 띠)
ROI_X1, ROI_X2 = 950, 970
ROI_Y1, ROI_Y2 = 158, 922

# 표준편차로 PCB 존재 여부 판단 (BGR 사용 - 구분력 높음)
roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
std = np.std(roi)  # BGR 그대로 계산

# BGR vs Gray 표준편차 비교:
# - PCB: BGR ~45-47, Gray ~27-30
# - 배경: BGR ~3, Gray ~3
# → BGR이 구분력 ~15배, Gray는 ~9배
```

### 2. 크롭

```python
# 배경 빼기로 PCB 영역 추출
diff = cv2.absdiff(gray, background_gray)
_, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

# 조건에 맞는 컨투어 찾기 (크기 + 경계 체크)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if validate_size(w, h) and is_fully_in_frame(x, w):
        return frame[y:y+h, x:x+w]
```

### 3. 크기 검증

```python
# PCB 크기 범위 확인
750 <= h <= 780
700 <= w <= 1600
0.8 <= w/h <= 2.0  # 실제 PCB 비율 ~1.9
```

### 4. 경계 체크

```python
# PCB가 프레임 안에 완전히 들어왔는지 확인
margin = 10
if x < margin or (x + w) > (frame_w - margin):
    return None  # 아직 가장자리에 걸쳐 있음
```

## 배경 이미지 생성

### 방법: 스트립 타일링

RTSP 스트림에서 직접 배경을 캡처합니다.

```bash
uv run python capture_background.py
```

**동작 원리:**
1. 가운데 스트립 (x=955~965, 10px 너비) 모니터링
2. 표준편차 < 15 (PCB 없음) 감지
3. 해당 스트립 캡처 (10px × 1080px)
4. 가로로 타일링해서 1920×1080 배경 생성

**왜 이 방법?**
- PCB 사이 간격이 좁아도 가운데만 비면 캡처 가능
- 전체 프레임이 비는 순간을 기다릴 필요 없음
- 컨베이어 벨트가 균일하므로 타일링해도 자연스러움

## RTSP 서버 정보

- **URL**: `rtsp://3.36.185.146:8554/pcb_stream`
- **위치**: Lightsail
- **해상도**: 1920×1080 @ 30fps
- **코덱**: H.264

## 백엔드 연동

크롭된 PCB 이미지는 추론 후 백엔드로 전송:

```python
POST http://3.35.182.98:8080/detect
{
  "timestamp": "2026-01-18T15:01:00",
  "image_id": "PCB_002",
  "image": "base64_encoded_string",  # 불량일 때만
  "detections": [...]
}
```

## 테스트

### 배경 캡처

```bash
uv run python capture_background.py
```

### 로컬 테스트 (mp4 파일)

```bash
uv run python main.py --input ../rtsp/PCB_Conveyor_30fps.mp4 --loop
```

### RTSP 테스트

```bash
uv run python main.py
```

### 디버그 모드 (크롭 이미지 저장)

```bash
uv run python main.py --debug --max-crops 5
```

## 주의사항

- **Queue 크기**: frame_queue=2, crop_queue=10
- **프레임 드롭**: Queue 가득 차면 오래된 프레임 버림 (실시간성 우선)
- **상태 초기화**: PCB가 나가면 (background 전환 시) crop_done 리셋
- **배경 이미지**: 반드시 실제 스트림에서 캡처해야 함 (생성 X)
- **RTSP 연결**: TCP 모드 필수, 연결 실패 시 자동 재시도

## 트러블슈팅

### RTSP 프레임 0개

1. TCP 모드 확인: `os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"`
2. 연결 재시도 로직 확인
3. VLC로 스트림 테스트: `vlc rtsp://3.36.185.146:8554/pcb_stream`

### 크롭이 잘림

1. 배경 이미지에 PCB가 포함되어 있는지 확인
2. `capture_background.py`로 새 배경 캡처
3. 경계 체크 마진 확인 (현재 10px)

### 크롭이 안 됨

1. 크기 검증 범위 확인 (w: 700-1600, h: 750-780, ratio: 0.8-2.0)
2. 디버그 로그 추가해서 bbox 좌표 확인
3. 배경 빼기 임계값 조정 (현재 25)

### 엔진 빌드 중 멈춤/지연

1. **원인**: TensorRT 엔진(.engine) 생성은 CPU/GPU를 풀가동하며 10~20분 소요됨
2. **해결**: 첫 실행 시 로그가 멈춘 것처럼 보여도 완료될 때까지 대기
3. **최적화**: 현재 FP16(`half=True`) 옵션이 적용되어 있어 빌드 후 성능 극대화

### 엔진 빌드 중 RTSP 에러 (Bad CSeq 등)

1. **원인**: 엔진 빌드 시 리소스 부족으로 네트워크 패킷 처리가 늦어짐
2. **해결**: `main.py` 수정으로 엔진 초기화를 먼저 수행하도록 개선됨

## 코드 컨벤션

- 주석은 한국어로 작성
- 커밋 메시지: `[Edge] 타입: 제목`
