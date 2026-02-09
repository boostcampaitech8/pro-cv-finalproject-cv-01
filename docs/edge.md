# 엣지 (Jetson)

## 개요

PCB 결함 탐지 시스템의 엣지 전처리 파이프라인입니다.

**역할:** RTSP 영상 수신 → PCB 감지/크롭 → 추론 → 백엔드 전송

## 기술 스택

| 구분 | 기술 |
|------|------|
| Python | >=3.11 |
| OpenCV | 영상 처리 |
| NumPy | 수치 연산 |
| Threading/Queue | 비동기 파이프라인 |

## 파이프라인 구조

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ RTSP 수신   │────→│  전처리      │────→│ 추론 워커   │
│ (Thread)    │     │  (Main)      │     │ (Thread)    │
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │                    │
   frame_queue        crop_queue           결과 전송
```

## 프로젝트 구조

```
serving/edge/
├── main.py               # 메인 루프 (통합)
├── preprocessor.py       # 전처리 파이프라인 클래스
├── rtsp_receiver.py      # RTSP 수신 스레드
├── capture_background.py # 배경 이미지 캡처 도구
├── background.png        # 생성된 배경 이미지
├── pyproject.toml
├── uv.lock
└── CLAUDE.md             # 상세 가이드
```

## 실행 방법

```bash
cd serving/edge
uv sync --active
uv run python main.py
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | RTSP URL 또는 비디오 파일 | `rtsp://3.36.185.146:8554/pcb_stream` |
| `--loop`, `-l` | 비디오 파일 반복 재생 | False |
| `--debug`, `-d` | 디버그 모드 (크롭 저장) | False |
| `--max-crops` | 최대 크롭 개수 (0=무제한) | 0 |

## 핵심 클래스

### PCBPreprocessor

프레임을 처리하여 PCB를 감지하고 크롭하는 클래스입니다.

**상태 머신:**
- `background`: 배경 상태 (PCB 없음)
- `pcb`: PCB 진입 상태

**히스테리시스 임계값:**
- `THRESH_LOW = 15`: 이 이하면 background로 전환
- `THRESH_HIGH = 25`: 이 이상이면 pcb로 전환

### RTSPReceiver

RTSP 스트림을 수신하는 데몬 스레드입니다.

**특징:**
- TCP 모드 사용
- 연속 실패 30회까지 재시도
- Queue가 가득 차면 오래된 프레임 버림
- graceful shutdown 지원

## 전처리 로직

### 1. PCB 진입 감지

```python
# 검사 영역 (세로 띠)
ROI_X1, ROI_X2 = 950, 970
ROI_Y1, ROI_Y2 = 158, 922

# 표준편차로 PCB 존재 여부 판단
roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
std = np.std(roi)  # BGR 사용
```

### 2. 크롭

```python
# 배경 빼기로 PCB 영역 추출
diff = cv2.absdiff(gray, background_gray)
_, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

# 조건에 맞는 컨투어 찾기
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
0.8 <= w/h <= 2.0
```

### 4. 경계 체크

```python
# PCB가 프레임 안에 완전히 들어왔는지 확인
margin = 10
if x < margin or (x + w) > (frame_w - margin):
    return None
```

## 배경 이미지 생성

### 스트립 타일링 방식

RTSP 스트림에서 직접 배경을 캡처합니다.

```bash
uv run python capture_background.py
```

**동작 원리:**
1. 가운데 스트립 (x=955~965, 10px 너비) 모니터링
2. 표준편차 < 15 (PCB 없음) 감지
3. 해당 스트립 캡처 (10px × 1080px)
4. 가로로 타일링해서 1920×1080 배경 생성

**장점:**
- PCB 사이 간격이 좁아도 가운데만 비면 캡처 가능
- 전체 프레임이 비는 순간을 기다릴 필요 없음
- 컨베이어 벨트가 균일하므로 타일링해도 자연스러움

## RTSP 서버 정보

| 항목 | 내용 |
|------|------|
| URL | `rtsp://3.36.185.146:8554/pcb_stream` |
| 위치 | Lightsail (3.36.185.146) |
| 해상도 | 1920×1080 @ 30fps |
| 코덱 | H.264 |

## 백엔드 연동

크롭된 PCB 이미지는 추론 후 백엔드로 전송됩니다:

```python
POST http://3.35.182.98:8080/detect
{
  "timestamp": "2026-01-18T15:01:00",
  "image_id": "PCB_002",
  "image": "base64_encoded_string",
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

### 디버그 모드

```bash
uv run python main.py --debug --max-crops 5
```

## 주의사항

- **Queue 크기**: frame_queue=2, crop_queue=10
- **프레임 드롭**: Queue 가득 차면 오래된 프레임 버림
- **상태 초기화**: PCB가 나가면 crop_done 리셋
- **배경 이미지**: 반드시 실제 스트림에서 캡처
- **RTSP 연결**: TCP 모드 필수, 연결 실패 시 자동 재시도

## 상세 문서

전체 상세 구현은 `serving/edge/CLAUDE.md` 참조.
