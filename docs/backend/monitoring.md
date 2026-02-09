# MLOps 모니터링

시스템 건강 상태 자동 평가 및 Slack 실시간 알림 시스템입니다.

## 헬스 모니터링 (GET /monitoring/health)

시스템 건강 상태를 종합적으로 평가하는 API입니다.

### 4-카테고리 임계값 알림 시스템

4가지 모니터링 지표와 각 임계값:

| 카테고리 | 지표 | Warning | Critical | 권장 조치 |
|---------|------|---------|----------|-----------|
| 불량률 | defect_rate (%) | ≥ 10.0 | ≥ 20.0 | 생산 라인 점검 |
| 평균 신뢰도 | avg_confidence | < 0.85 | < 0.75 | 모델 재학습 |
| 저신뢰도 비율 | low_confidence_ratio (%) | ≥ 20.0 | ≥ 40.0 | 모델 성능 점검 |
| PCB당 평균 결함 | avg_defects_per_item | ≥ 2.0 | ≥ 3.0 | 공정 품질 긴급 점검 |

### 시스템 상태 결정 로직

- `critical`: 1개 이상의 critical 알림이 존재
- `warning`: critical 알림은 없으나 warning 알림이 1개 이상 존재
- `healthy`: 알림 없음

### 신뢰도 분포 분석

모든 결함의 신뢰도를 4개 구간으로 분류:

| 구간 | 범위 | 의미 |
|------|------|------|
| high | 0.9 ≤ conf ≤ 1.0 | 높은 신뢰도 |
| medium | 0.8 ≤ conf < 0.9 | 중간 신뢰도 |
| low | 0.7 ≤ conf < 0.8 | 낮은 신뢰도 |
| very_low | conf < 0.7 | 매우 낮은 신뢰도 |

## 경량 알림 API (GET /monitoring/alerts)

프론트엔드 실시간 폴링에 최적화된 엔드포인트입니다.

**특징:**
- 알림 목록과 요약 정보만 반환
- 1~2초 간격 폴링에 적합
- 응답 크기 최소화

## Slack 실시간 알림

### 이벤트 기반 아키텍처

별도의 스케줄러나 폴링 없이, `POST /detect` 호출 시마다 이벤트 기반으로 동작:

```
POST /detect 수신
  → DB 저장
  → get_health() 호출
  → 이전 상태와 비교
  → 상태 변화 시에만 Slack 전송
  → 현재 상태를 메모리에 저장
```

### 상태 변화 감지

`last_status_per_session` 딕셔너리에 세션별 마지막 상태를 메모리에 저장합니다. 매 검사마다 현재 상태와 비교하여 변화가 감지된 경우에만 알림을 전송합니다.

### 6가지 상태 전환 메시지

| 전환 | 메시지 |
|------|--------|
| healthy → warning | ⚠️ 경고: 시스템 상태가 warning으로 전환 |
| healthy → critical | 🚨 긴급: 시스템 상태가 critical로 전환 |
| warning → critical | 🚨 악화: warning에서 critical로 악화 |
| warning → healthy | ✅ 개선: 정상 상태로 복구 |
| critical → warning | ⚠️ 부분 개선: critical에서 warning으로 |
| critical → healthy | ✅ 해결: 완전히 복구 |

### Slack 메시지 포맷

Slack Block Kit을 사용하여 구조화된 메시지를 전송합니다.

**예시:**

```
🚨 악화: 상태가 warning에서 critical로 악화되었습니다
──────────────────
상태: critical          세션: 세션 1 (활성)

🔴 Critical 알림:
• 불량률이 25.0%로 매우 높습니다
  → 생산 라인 점검 및 원인 분석 필요
```

**특징:**
- 비동기 HTTP 클라이언트(httpx.AsyncClient) 사용
- 5초 타임아웃 설정
- 전송 실패 시에도 메인 플로우에 영향 없음

### 환경변수 설정

| 변수 | 기본값 | 설명 |
|------|--------|------|
| SLACK_WEBHOOK_URL | "" | Slack Incoming Webhook URL |
| SLACK_ALERT_ENABLED | false | 알림 활성화 여부 |

두 값 모두 설정되어야 알림이 활성화됩니다.

## 관련 파일

- `routers/detect.py`: `check_and_send_slack_alert()` - 상태 변화 감지 로직
- `utils/slack_notifier.py`: `send_slack_alert()` - 웹훅 전송 로직
- `config/settings.py`: 환경 변수 로드, 알림 임계값 설정
- `database/db.py`: `get_health()` - 건강 상태 조회
