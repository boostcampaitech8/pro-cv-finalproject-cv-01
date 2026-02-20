# Edge Multi-Camera Issue Tickets

기준 문서: `docs/edge_multicam_improvement_guide.md`  
목적: 팀 공유 및 스프린트 실행용 작업 티켓 정리

## 진행 현황

- [x] P0-1 카메라별 전처리 상태 분리
- [x] P0-2 `camera_id` API/DB 저장 일관화
- [x] P0-3 ID/실패 파일명 충돌 방지
- [x] P0-4 모델 경로 인자 버그 수정

## 사용 방법

1. 각 티켓은 독립 PR 단위로 진행한다.
2. 완료 시 DoD(Definition of Done)와 검증 로그를 티켓에 첨부한다.
3. P0 -> P1 -> P2 순서로 처리한다.

---

## Ticket P0-1: 카메라별 전처리 상태 분리

- 우선순위: P0
- 목표:
  - 카메라 간 전처리 상태 오염 제거
- 범위:
  - `serving/edge/main.py`
  - `serving/edge/preprocessor.py`
- 작업:
  - `camera_id -> PCBPreprocessor` 매핑 도입
  - 단일 전처리 인스턴스 공유 제거
- DoD:
  - 카메라별 상태가 독립적으로 동작
  - 독립성 테스트에서 상호 간섭 없음
- 검증:
  - cam1=pcb, cam2=background 입력 시 오검출/미검출 여부 확인

## Ticket P0-2: `camera_id` API/DB 저장 일관화

- 우선순위: P0
- 목표:
  - `camera_id`를 end-to-end로 추적 가능하게 구성
- 범위:
  - `serving/api/schemas/schemas.py`
  - `serving/api/database/db.py`
  - 관련 라우터/조회 API
- 작업:
  - DetectRequest에 `camera_id` 필드 추가
  - `inspection_logs`에 `camera_id` 컬럼 및 마이그레이션 반영
  - 조회 API에 `camera_id` 포함
- DoD:
  - DB에 camera_id 저장 확인
  - API 응답/통계에서 camera_id 필터 가능
- 검증:
  - 다중 카메라 전송 후 DB row별 camera_id 검증

## Ticket P0-3: ID/실패 파일명 충돌 방지

- 우선순위: P0
- 목표:
  - 동일 초 다건 처리 시 충돌 제거
- 범위:
  - `serving/edge/inference_worker.py`
  - `serving/edge/upload_worker.py`
- 작업:
  - `image_id` 생성 로직 UUID 또는 마이크로초+카운터로 교체
  - 실패 JSON 파일명 충돌 방지
- DoD:
  - 중복 파일 생성/덮어쓰기 없음
- 검증:
  - 고속 입력에서 수천 건 생성 후 이름 중복 0건

## Ticket P0-4: 모델 경로 인자 버그 수정

- 우선순위: P0
- 목표:
  - CLI 전달 모델 경로를 실제 로딩에 반영
- 범위:
  - `serving/edge/inference_worker.py`
- 작업:
  - `config.MODEL_PATH` 하드코딩 제거
  - 전달받은 `model_path` 사용
- DoD:
  - 다른 모델 경로 인자 전달 시 해당 모델 로드
- 검증:
  - 존재/부재 경로 각각 동작 확인

---

## Ticket P1-1: 카메라별 frame queue 분리 + 공정 스케줄링

- 우선순위: P1
- 목표:
  - 공유 큐 경쟁으로 인한 카메라 편향 완화
- 범위:
  - `serving/edge/main.py`
  - `serving/edge/rtsp_receiver.py`
- 작업:
  - camera별 frame queue 분리
  - 중앙 소비 루프에 라운드로빈 또는 가중치 스케줄러 적용
- DoD:
  - 카메라별 drop rate 편차 감소
- 검증:
  - 불균형 입력 부하에서 camera별 drop rate 비교

## Ticket P1-2: 큐 크기 스케일링 및 워터마크 경보

- 우선순위: P1
- 목표:
  - 카메라 수 증가 시 병목 지점 조기 탐지
- 범위:
  - `serving/edge/main.py`
  - 필요 시 공용 metrics 모듈
- 작업:
  - `crop/upload` 큐 동적 크기 정책
  - queue depth high watermark 경보 로그
- DoD:
  - 큐 적체 상황을 로그로 즉시 파악 가능
- 검증:
  - `-n 3+`에서 큐 적체 재현 시 경보 확인

## Ticket P1-3: Scavenger Worker + 재전송 정책

- 우선순위: P1
- 목표:
  - 네트워크 장애 시 데이터 유실 0%에 근접
- 범위:
  - `serving/edge/upload_worker.py`
  - 신규 `serving/edge/scavenger_worker.py` (필요 시)
- 작업:
  - 실패 파일 백그라운드 재전송
  - exponential backoff + jitter + max retry/TTL
- DoD:
  - 장애 복구 후 backlog 자동 해소
- 검증:
  - 네트워크 차단/복구 시나리오로 backlog 0 복구 시간 측정

## Ticket P1-4: RTSP 재연결 강화

- 우선순위: P1
- 목표:
  - 채널 단절 시 자동 복구
- 범위:
  - `serving/edge/rtsp_receiver.py`
- 작업:
  - 재연결 루프 및 백오프 추가
  - 채널 상태 변화 로그 표준화
- DoD:
  - 일시 단절 후 자동 복구
- 검증:
  - 채널 중단 후 복구 성공률 측정

## Ticket P1-E1: GStreamer + nvv4l2decoder 하드웨어 디코드 전환

- 우선순위: P1
- 목표:
  - RTSP 디코드 경로를 HW decode로 전환해 EMC/CPU 부하를 낮춘다.
- 범위:
  - `serving/edge/rtsp_receiver.py`
  - 필요 시 설정 파일(파이프라인 문자열 구성)
- 작업:
  - OpenCV 기본 FFMPEG 경로 대신 GStreamer 파이프라인 사용
  - `nvv4l2decoder` 기반 수신 파이프라인 적용
  - 기존 경로 대비 fallback 옵션 유지
- DoD:
  - `-n 3` 기준 EMC load, CPU 사용률, drop rate 개선 확인
- 검증:
  - before/after 동일 시나리오에서 `tegrastats` + edge 로그 비교

## Ticket P1-E2: GPU 전처리 POC 및 채택 여부 결정

- 우선순위: P1 (E1 이후)
- 목표:
  - 전처리 GPU화가 실제 E2E 성능에 이득이 있는지 검증
- 범위:
  - `serving/edge/preprocessor.py`
  - 필요 시 신규 GPU 전처리 모듈
- 작업:
  - GPU 전처리 POC 구현 (`cv2.cuda` 또는 NV 경로)
  - CPU<->GPU 메모리 복사 비용 포함 계측
  - E2E p95, drop, EMC 기준으로 채택/보류 결정
- DoD:
  - POC 결과와 채택 여부가 수치 근거로 문서화
- 검증:
  - 동일 입력에서 CPU 전처리 vs GPU 전처리 A/B 비교

---

## Ticket P2-1: 의존성 명세 정리

- 우선순위: P2
- 목표:
  - 로컬/컨테이너 재현성 확보
- 범위:
  - `serving/edge/pyproject.toml`
  - `serving/edge/uv.lock`
- 작업:
  - 누락 의존성 반영
  - lock 갱신
- DoD:
  - 신규 환경에서 설치/실행 가능
- 검증:
  - 클린 환경 설치 테스트

## Ticket P2-2: RTSP 스트리머 관리 스크립트 보강

- 우선순위: P2
- 목표:
  - 운영 중 스트리머 정리 안정성 향상
- 범위:
  - `serving/rtsp/stream.sh`
- 작업:
  - 패턴 삭제 의존성 줄이고 명시 삭제로 개선
- DoD:
  - 실행/재실행 시 의도한 프로세스만 관리
- 검증:
  - 반복 실행 테스트에서 orphan process 0건

---

## 성능 계측 티켓

## Ticket M-1: 단계별 성능 지표 수집 코드 반영

- 우선순위: P0-P1 병행
- 목표:
  - 성능 지표를 수치로 수집 가능하게 구성
- 지표:
  - Input FPS (camera별)
  - Processing FPS (camera별/전체)
  - Inference Throughput (crops/sec)
  - Drop Rate (camera별)
  - Preprocess/Inference/QueueWait/Upload/E2E Latency p50/p95
  - Upload Success Rate + Backlog
  - CPU/GPU/메모리/온도
  - EMC load 및 throttling 이벤트 수
- 범위:
  - `serving/edge/main.py`
  - `serving/edge/rtsp_receiver.py`
  - `serving/edge/inference_worker.py`
  - `serving/edge/upload_worker.py`
- DoD:
  - 5초 주기 요약 로그 + 종료 시 전체 통계 출력
- 검증:
  - `-n 1`, `-n 3`에서 지표 누락 없이 기록

## Ticket M-2: Before/After 벤치마크 리포트

- 우선순위: P1 후
- 목표:
  - 개선 효과를 수치로 입증
- 시나리오:
  - `-n 1`, `-n 3`, 스케일 테스트, 장애 복구 테스트
- 산출물:
  - 비교표 (before vs after)
  - bottleneck 분석
  - 채널 수 권장안
- DoD:
  - 팀 리뷰 가능한 리포트 1부 완료

---

## 스프린트 제안 순서

1. P0-1, P0-2, P0-3, P0-4
2. M-1 (계측)
3. P1-1, P1-2, P1-3, P1-4, P1-E1, P1-E2
4. M-2 (before/after 리포트)
5. P2-1, P2-2
