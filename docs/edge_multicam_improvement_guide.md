# Edge Multi-Camera Improvement Guide

작성일: 2026-02-20  
적용 범위: `serving/edge`, `serving/rtsp`, `serving/api`

## 1. 목적

이 문서는 Edge 멀티카메라 개선 작업의 기준 문서다.  
향후 코드 수정, 성능 검증, 팀 공유는 본 문서를 기준으로 진행한다.

## 2. 현재 구조 요약

현재 구조는 다음과 같다.

1. RTSP 수신은 카메라 수만큼 병렬 스레드
2. 전처리는 `main.py` 루프 1개에서 순차 처리
3. 추론은 `InferenceWorker` 1개에서 순차 처리
4. 업로드는 `UploadWorker` 1개에서 순차 처리
5. `frame/crop/upload` 큐는 공유

즉, "수신 병렬 + 후단 순차 + 공유 큐" 구조다.

## 3. 고정 원칙

아래 원칙은 변경 전 합의가 필요하다.

1. 카메라별 수신/전처리 상태는 독립이어야 한다.
2. 추론 모델은 기본적으로 1개만 로드한다.
3. `camera_id`는 Edge -> API -> DB -> 조회까지 일관되게 유지한다.
4. 전송 실패 데이터는 로컬 내구 저장 + 재전송 경로를 보장한다.

## 4. 권장 목표 구조

1. 카메라별 독립
- `RTSPReceiver` (camera_id별)
- `frame_queue` (camera_id별)
- `PCBPreprocessor` 인스턴스 (camera_id별)

2. 공용
- `InferenceWorker` (모델 1개)
- 공용 `crop_queue` (또는 공정 스케줄링 큐)
- `UploadWorker`
- `ScavengerWorker` (재전송)

3. 스케줄링
- 카메라별 입력이 불균형해도 특정 카메라가 굶지 않도록 라운드로빈/가중치 정책을 적용한다.

## 5. 문제점 및 해결 항목

## P0 (즉시)

1. 전처리 상태 공유
- 문제: 카메라 간 상태 오염 가능
- 해결: `camera_id -> PCBPreprocessor` 분리

2. `camera_id` 백엔드 저장 누락
- 문제: Edge payload에는 있으나 API/DB 반영 불완전
- 해결: API 스키마 + DB 컬럼 + 조회 API 반영

3. `image_id`/실패 파일명 충돌 위험
- 문제: 초 단위 식별자로 중복 가능
- 해결: UUID 또는 마이크로초+증분 카운터 사용

4. `model_path` 인자 사용 버그
- 문제: 전달 인자 대신 전역 경로 사용
- 해결: 인자로 전달된 경로를 실제 로딩 경로로 사용

## P1 (성능/안정성)

5. 공유 `frame_queue` 경쟁
- 문제: 한 카메라 과점 시 타 카메라 드롭 증가
- 해결: 카메라별 frame queue 분리 + 중앙 소비 스케줄링

6. 큐 크기 스케일링 미흡
- 문제: `crop/upload` 큐 병목 가능
- 해결: 카메라 수/이벤트율 기반 동적 조정

7. 재전송 워커 부재
- 문제: 실패 저장만 있고 자동 복구 경로 부족
- 해결: Scavenger + backoff + jitter + 최대 재시도

8. RTSP 재연결 내구성 강화 필요
- 문제: 실패 누적 시 채널 종료
- 해결: 재연결 루프 + 백오프 + 채널 상태 리포트

9. EMC load 과다 및 over-current throttling 대응 미흡
- 문제: 멀티 RTSP 입력 시 메모리 대역폭(EMC) 과점과 전류 피크로 성능 저하 가능
- 해결:
  - 우선순위 1: RTSP 수신을 GStreamer + `nvv4l2decoder` 기반 하드웨어 디코드로 전환
  - 우선순위 2: CPU 메모리 왕복 복사를 줄이는 경로로 전처리 파이프라인 재설계
  - 우선순위 3: 필요 시 GPU 전처리(`cv2.cuda` 또는 NV 경로) POC 수행 후 채택

## P2 (재현성/운영)

9. 의존성 명세 누락
- 문제: 런타임 패키지 일부 미기재
- 해결: `pyproject.toml` 정리 및 lock 갱신

10. RTSP 스크립트 정리 로직 보강
- 문제: 패턴 삭제 명령의 환경 의존성
- 해결: `pm2 ls` 기반 명시 삭제 방식으로 변경

## 6. 성능 지표 기준 (필수 반영)

## 6.1 지표 정의

1. `Input FPS (camera별)`
- 수신 스레드가 실제 읽은 FPS

2. `Processing FPS`
- 전처리 루프가 처리한 FPS
- 주의: 현재 `main.py`의 평균 FPS는 전체 합산 FPS다.

3. `Inference Throughput (crops/sec)`
- 초당 추론 처리 crop 수

4. `Drop Rate (camera별)`
- `drop_count / frame_count`

5. `Preprocess Latency p50/p95 (ms)`
- 전처리 1건 처리시간

6. `Inference Latency p50/p95 (ms)`
- 추론 1건 처리시간

7. `Queue Wait p50/p95 (ms)`
- `frame/crop/upload` 큐 대기시간

8. `Upload Latency p50/p95 (ms)`
- API 전송 왕복 시간

9. `E2E Latency p50/p95 (ms)`
- frame 수신 시점부터 업로드 성공까지 총 지연

10. `Upload Success Rate + Backlog`
- 성공률/실패율, `storage/failed` 증가량, 복구시간

11. `Resource Metrics`
- CPU/GPU/메모리/온도 (`tegrastats`)

12. `EMC/Throttle Metrics`
- EMC load (%)
- throttling 이벤트 개수 (`System throttled due to over-current` 포함)
- 이벤트 발생 시점의 FPS/drop/latency 동시 기록

## 6.2 p50/p95 해석 규칙

1. `p50`: 중앙값, 일반 상태 성능
2. `p95`: 느린 5% 포함, 병목/튀는 지연 확인
3. 운영 판단은 평균보다 `p95`를 우선한다.

## 6.4 EMC 판단 규칙

1. EMC load가 장시간 고점(예: 90%+ 구간 지속)이고 drop/e2e p95가 함께 악화되면 메모리 대역폭 병목으로 판정한다.
2. over-current throttling 로그가 반복 발생하면 성능 이슈를 전력/전류 피크 이슈와 함께 다룬다.
3. 개선 우선순위는 `HW decode -> memory copy 감소 -> GPU 전처리` 순으로 적용한다.

## 6.3 합격 기준 예시

아래는 초기 기준이며 실제 값은 테스트 후 팀 합의로 확정한다.

1. Drop Rate(camera별) <= 1%
2. Inference p95가 목표 공정 간격 내 유지
3. E2E p95가 현장 응답 시간 SLA 이내
4. Upload Success Rate >= 99%
5. 장애 복구 후 backlog가 제한 시간 내 0으로 복구

## 7. 테스트 방법 (Before/After 공통)

## 7.1 고정 조건

1. 동일 Jetson 장비
2. 동일 모델/배경/입력 소스
3. 동일 실행 인자
4. 각 시나리오 3회 반복, 10~30분 측정

## 7.2 시나리오

1. `-n 1` 단일 카메라 안정성
2. `-n 3` 멀티카메라 정상 부하
3. `-n 1 -> 2 -> 3 -> ...` 스케일 테스트
4. 카메라별 상이 입력 독립성 테스트
5. 네트워크 차단/복구 재전송 테스트
6. 한 채널 장애 격리 테스트

## 8. 작업 순서

1. P0 전부 처리
2. P1 처리
3. P2 처리
4. 성능 재측정
5. 도커/컴포즈 확장

## 9. 도커/컴포즈 적용 기준

1. 먼저 단일 Edge 컨테이너(모델 1개 공유) 안정화
2. 이후 운영 편의 목적으로 Compose 적용
3. 카메라별 추론 컨테이너 복제는 기본 전략으로 채택하지 않음

## 10. 체크리스트

- [x] P0-1 전처리 카메라별 분리
- [x] P0-2 `camera_id` API/DB 반영
- [x] P0-3 ID/실패 파일 충돌 방지
- [x] P0-4 model_path 인자 버그 수정
- [ ] P1-5 frame queue 분리 + 스케줄링
- [ ] P1-6 queue 스케일링
- [ ] P1-7 Scavenger + backoff
- [ ] P1-8 RTSP 재연결 강화
- [ ] P1-E1 GStreamer + nvv4l2decoder HW decode 전환
- [ ] P1-E2 GPU 전처리 POC (복사 비용 포함 성능 검증)
- [ ] P2-9 edge 의존성 정리
- [ ] P2-10 RTSP 스크립트 보강
- [ ] 성능 지표 수집 코드 반영
- [ ] Before/After 벤치마크 리포트 작성
