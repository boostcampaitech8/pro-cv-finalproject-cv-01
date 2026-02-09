# 🍇 PODO - PCB Only Detected Once

> PCB 기판의 결함을 실시간으로 탐지하는 Edge AI 시스템

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

## 프로젝트 소개

제조 현장에서 PCB 기판의 결함(scratch, hole 등)을 사람이 육안으로 검수하는 것은 느리고 일관성이 떨어집니다.
**PODO**는 컨베이어 벨트 위를 지나는 PCB를 실시간으로 촬영하고, Edge AI(Jetson Orin Nano)에서 YOLOv11 모델로 즉시 추론하여 결함을 자동 탐지합니다.
탐지 결과는 백엔드 서버와 AWS Storage에 저장되고, 대시보드를 통해 실시간 모니터링할 수 있습니다.

**핵심 기능**
- RTSP 영상 스트림에서 PCB를 자동 감지하고 크롭하여 추론
- YOLOv8 + QAT 경량화 모델로 Edge 디바이스에서 실시간 결함 탐지
- 불량률, 신뢰도 분포, 결함 유형별 통계를 제공하는 실시간 대시보드
- 임계값 기반 알림 시스템 (Slack 연동)
- 품질 관리자 피드백 → 자동 재라벨링 → S3 저장으로 이어지는 MLOps 파이프라인

**핵심 성과**

| 지표 | 수치 | 설명 |
|------|------|------|
| mAP@50 | **96.0%** | 거의 모든 결함을 정확히 탐지 |
| Recall | **100%** | 불량을 단 하나도 놓치지 않음 |
| 추론 속도 | **33.8 FPS** | 30fps 영상의 모든 프레임을 실시간 처리 |
| 경량화 가속 | **2.4x** | PyTorch Native 대비 TensorRT QAT INT8 |


## 데모 영상
> ⚠️ 영상을 누르면 재생 됩니다.

[![데모 영상](https://img.youtube.com/vi/7DkqRYQfxBg/0.jpg)](https://youtu.be/7DkqRYQfxBg?si=3mS4f99RSzwpcFdx)


### **라이브 데모**: http://3.35.182.98/ 

> ⚠️ 서버는 2026년 3월 말까지 운영 예정입니다.

### Dashboard Screenshot
![대시보드](docs/images/dashboard.png)


## 시스템 아키텍처
### System Overview
![오버뷰](docs/images/overview.png)


### System Architecture
![아키텍처](docs/images/architecture.png)




**데이터 흐름:**
1. **RTSP 서버** (Lightsail) — PCB 영상을 RTSP 프로토콜로 스트리밍
2. **엣지** (Jetson Orin) — RTSP 수신 → 배경 차분으로 PCB 크롭 → YOLO 추론 → 결과 전송
3. **백엔드** (EC2 / FastAPI) — 추론 결과 및 이미지를 DB·S3에 저장, 통계 API 제공, Slack 알림
4. **프론트엔드** (EC2 / nginx) — 실시간 대시보드로 검사 현황 시각화 + 피드백 제출
5. **피드백 → 재라벨링** — 품질 관리자가 bbox별 피드백(오탐/미탐/클래스 수정)을 제출하면 수정된 YOLO 라벨이 S3 `refined/`에 자동 저장되어 모델 재학습에 활용

## 기술 스택

| 모듈 | 기술 |
|------|------|
| 백엔드 | FastAPI, SQLite (aiosqlite), Pydantic |
| 엣지 | Python, OpenCV, NumPy, Threading/Queue |
| 프론트엔드 | HTML/CSS/JS, Bootstrap (Paper Dashboard) |
| 학습 | YOLOv8, QAT (Quantization-Aware Training) |
| RTSP | MediaMTX, H.264 |
| MLOps | S3 (재라벨링 데이터셋), Slack Webhook (알림) |
| 인프라/CI | GitHub Actions, AWS (EC2, S3, Lightsail), systemd, nginx |

## 모델

모델 크기 3종(Nano/Small/Medium) x 해상도 4종(640~1600px) x 버전 2종(v8/v11), 총 **24가지 조합**을 실험하여 최적 모델을 선택했습니다.

**주요 실험 결과 (V100 기준):**

| 모델 | 해상도 | mAP50 | Recall | FPS | 비고 |
|------|--------|-------|--------|-----|------|
| YOLOv11n | 960px | 0.9662 | 1.0 | 44.79 | 가성비 우수 |
| **YOLOv11m** | **640px** | **0.9620** | **1.0** | **46.23** | **배포 표준** |
| YOLOv8s | 1280px | 0.9776 | 1.0 | 40.03 | 고성능 대안 |
| YOLOv11m | 960px | 0.9742 | 1.0 | 37.66 | 고성능 대안 |

**경량화 (Jetson Orin Nano 배포):**

| 단계 | 방식 | FPS | 지연 시간 | mAP50 | Recall |
|------|------|-----|----------|-------|--------|
| Step 1 | TensorRT FP16 | 30.7 | 32.6ms | 0.9586 | 1.0 |
| Step 2 | PTQ (INT8) | 36.8 | 27.2ms | 0.9487 | 1.0 |
| **Step 3** | **QAT (INT8)** | **33.8** | **29.6ms** | **0.9602** | **1.0** |

> QAT 자체 파이프라인 구축: Ultralytics 기본 export 불가 → Deep Recursive Injection + EMA Trainer + Re-calibration 직접 구현

**최종 선택: YOLOv11m + TensorRT QAT INT8**
- FP32 수준의 정밀도를 유지하며 추론 속도 극대화 (PyTorch 대비 약 2.4배)
- 33.8 FPS로 30fps 영상 소스의 모든 프레임 실시간 처리 가능


## 시작하기

### 사전 요구사항

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- (엣지) NVIDIA Jetson Orin + YOLO 모델 파일

### 백엔드

```bash
cd serving/api
uv sync --active
uv run uvicorn main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

### 엣지

```bash
cd serving/edge
uv sync --active
uv run python main.py --input rtsp://YOUR_RTSP_URL
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | RTSP URL 또는 비디오 파일 | RTSP 서버 주소 |
| `--loop`, `-l` | 비디오 파일 반복 재생 | False |
| `--debug`, `-d` | 디버그 모드 (크롭 저장) | False |

### 프론트엔드

`dashboard/` 디렉토리의 정적 파일을 nginx로 서빙합니다.

```bash
# nginx 설정에서 /api/* 를 백엔드로 프록시
sudo systemctl start nginx
```

## 프로젝트 구조

```
.
├── serving/
│   ├── api/                    # 백엔드 (FastAPI)
│   │   ├── main.py             #   앱 진입점
│   │   ├── routers/            #   API 라우터 (detect, stats, monitoring, feedback ...)
│   │   ├── database/           #   SQLite 연동
│   │   ├── schemas/            #   Pydantic 모델
│   │   ├── config/             #   설정 (Slack, 알림 임계값 등)
│   │   └── utils/              #   이미지 처리, Slack 알림
│   ├── edge/                   # 엣지 전처리 + 추론 (Jetson)
│   │   ├── main.py             #   파이프라인 메인 루프
│   │   ├── preprocessor.py     #   PCB 감지/크롭
│   │   ├── rtsp_receiver.py    #   RTSP 수신 스레드
│   │   ├── inference_worker.py #   추론 워커
│   │   └── upload_worker.py    #   결과 업로드 워커
│   └── rtsp/                   # RTSP 스트림 서버
├── dashboard/                  # 프론트엔드 (HTML/JS)
├── training/                   # 모델 학습 (YOLOv8 + QAT)
├── docs/                       # 문서
└── .github/workflows/          # CI/CD (GitHub Actions)
```

## 📚 상세 문서

각 모듈의 상세 구현 문서는 `docs/` 폴더에 있습니다.

| 모듈 | 문서 |
|------|------|
| 🎨 프론트엔드 | [docs/frontend.md](docs/frontend.md) |
| ⚙️ 백엔드 | [docs/backend/](docs/backend/) |
| 🔧 엣지 | [docs/edge.md](docs/edge.md) |
| 🧠 학습 | [docs/training.md](docs/training.md) |
| 📡 RTSP | [docs/rtsp.md](docs/rtsp.md) |

---

## API 명세

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/detect` | 엣지 추론 결과 수신 (다중 결함 지원) |
| GET | `/stats` | 검사 통계 (불량률, 신뢰도, FPS 등) |
| GET | `/latest` | 최근 검사 이력 10건 |
| GET | `/defects` | 결함 타입별 집계 |
| GET | `/monitoring/health` | 시스템 건강 상태 (세션별 필터링) |
| GET | `/monitoring/alerts` | 알림 조회 (프론트엔드 폴링용) |
| POST | `/feedback/bulk` | 다중 bbox 피드백 + 자동 재라벨링 |
| GET | `/feedback/stats` | 피드백 통계 (MLOps) |
| GET | `/feedback/queue` | 라벨링 대기열 조회 |
| GET | `/feedback/export` | 재학습용 데이터셋 정보 (S3 경로) |

**POST /detect 요청 예시:**

```json
{
  "timestamp": "2026-01-18T15:01:00",
  "image_id": "PCB_002",
  "image": "base64_encoded_string",
  "detections": [
    {"defect_type": "scratch", "confidence": 0.95, "bbox": [10, 20, 100, 120]},
    {"defect_type": "hole", "confidence": 0.92, "bbox": [300, 350, 320, 380]}
  ]
}
```

## 👥 팀 소개

<table>
<tr>
<td align="center" width="20%" style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px;">
<img src="docs/images/team/kyungmo.png" width="300px" alt="김경모"/>
<br />
<b>김경모</b>
<br />
<code>Frontend</code> <code>Modeling</code>
<br />
<sub>JavaScript 기반 웹 대시보드 구축, MLOps 구현, 모델 경량화</sub>
</td>
<td align="center" width="20%" style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px;">
<img src="docs/images/team/jieun.png" width="300px" alt="김지은"/>
<br />
<b>김지은</b>
<br />
<code>Backend</code> <code>MLOps</code> <code>CI/CD</code> <code>AWS</code>
<br />
<sub>FastAPI/DB 설계 및 구축, MLOps 기능 설계 및 구축, GitHub Actions 기반 CI/CD, AWS EC2 관리</sub>
</td>
<td align="center" width="20%" style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px;">
<img src="docs/images/team/jungho.png" width="300px" alt="위정호"/>
<br />
<b>위정호</b>
<br />
<code>Modeling</code> <code>AWS</code>
<br />
<sub>YOLO 기반 모델링 및 경량화, AWS S3 관리</sub>
</td>
<td align="center" width="20%" style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px;">
<img src="docs/images/team/bonghak.png" width="300px" alt="이봉학"/>
<br />
<b>이봉학</b>
<br />
<code>Edge Device</code>
<br />
<sub>Jetson Orin Nano 기반 환경 설정 및 추론 시스템 구축</sub>
</td>
<td align="center" width="20%" style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px;">
<img src="docs/images/team/subin.png" width="300px" alt="조수빈"/>
<br />
<b>조수빈</b>
<br />
<code>PM</code> <code>AWS</code> <code>RTSP</code>
<br />
<sub>프로젝트 관리, 네트워크 보안 설정, RTSP 영상 제작 및 송출, AWS lightsail 관리</sub>
</td>
</tr>
</table>

## 라이선스

이 프로젝트는 부스트캠프 AI Tech 과정에서 제작되었습니다.
