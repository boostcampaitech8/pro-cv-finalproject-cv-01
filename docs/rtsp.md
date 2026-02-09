# RTSP 서버

PCB 검사용 카메라 영상을 RTSP 프로토콜로 스트리밍하는 서버입니다.

## 개요

**역할:** PCB 검사 영상을 실시간 스트리밍

## 기술 스택

| 구분 | 기술 |
|------|------|
| 스트리밍 서버 | MediaMTX |
| 영상 처리 | Python, OpenCV |
| 코덱 | H.264 |

## 프로젝트 구조

```
serving/rtsp/
├── main.py               # RTSP 스트림 전송 스크립트
├── PCB_Conveyor_30fps.mp4 # 테스트용 영상
├── pyproject.toml
└── uv.lock
```

## 서버 정보

| 항목 | 내용 |
|------|------|
| 위치 | Lightsail (3.36.185.146) |
| URL | `rtsp://3.36.185.146:8554/pcb_stream` |
| 포트 | 8554 |
| 해상도 | 1920×1080 @ 30fps |
| 코덱 | H.264 |

## 실행 방법

### 로컬 테스트

```bash
cd serving/rtsp
uv sync
uv run python main.py
```

### 서버 배포

Lightsail 서버에 MediaMTX 설치 및 실행:

```bash
# MediaMTX 설치
wget https://github.com/bluenviron/mediamtx/releases/download/v1.3.0/mediamtx_v1.3.0_linux_amd64.tar.gz
tar -xzf mediamtx_v1.3.0_linux_amd64.tar.gz

# 실행
./mediamtx
```

## 테스트

### VLC로 스트림 확인

```bash
vlc rtsp://3.36.185.146:8554/pcb_stream
```

### FFmpeg로 스트림 확인

```bash
ffplay rtsp://3.36.185.146:8554/pcb_stream
```

### Python OpenCV로 스트림 확인

```python
import cv2

cap = cv2.VideoCapture('rtsp://3.36.185.146:8554/pcb_stream')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('RTSP Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## 클라이언트 연동

### 엣지 (Jetson)

엣지 모듈에서 RTSP 스트림을 수신합니다:

```bash
cd serving/edge
uv run python main.py --input rtsp://3.36.185.146:8554/pcb_stream
```

## 트러블슈팅

### 스트림 연결 실패

1. **방화벽 확인**: 포트 8554가 열려 있는지 확인
2. **서버 상태 확인**: MediaMTX 프로세스가 실행 중인지 확인
3. **네트워크 확인**: ping으로 서버 연결 확인

### 프레임 드롭

1. **TCP 모드 사용**: UDP 대신 TCP 모드 사용 (안정성 향상)
2. **버퍼 크기 조정**: OpenCV VideoCapture 버퍼 크기 조정
3. **네트워크 대역폭 확인**: 네트워크 속도가 충분한지 확인

## 참고 자료

- [MediaMTX GitHub](https://github.com/bluenviron/mediamtx)
- [RTSP 프로토콜 RFC](https://tools.ietf.org/html/rfc2326)
