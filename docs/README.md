
# 구현 계획 문서

PCB 결함 탐지 시스템의 파트별 구현 계획을 관리합니다.

## 문서 목록

- [backend.md](backend.md) - FastAPI 백엔드 서버
- [frontend.md](frontend.md) - 대시보드
- [edge.md](edge.md) - Jetson 추론/전송
- [training.md](training.md) - YOLO 학습

## 시스템 아키텍처
```
┌──────────┐                ┌───────────┐               ┌──────────┐
│  Jetson  │  POST /detect  │  FastAPI  │  GET /stats   │Streamlit │
│  (엣지)  │ ─────────────→ │ (백엔드)  │ ←──────────── │ (프론트) │
└──────────┘                └───────────┘               └──────────┘
```

## 커밋 컨벤션

`.gitmessage` 템플릿 사용:
```
[파트] 타입: 제목

예시:
[BE] feat: POST /detect 엔드포인트 구현
[BE] fix: Base64 디코딩 오류 수정
```