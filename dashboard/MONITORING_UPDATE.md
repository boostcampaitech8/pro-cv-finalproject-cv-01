# Dashboard & Monitoring System Update Summary

이번 업데이트를 통해 PCB 결함 탐지 시스템의 시각화 대시보드와 시스템 건전성 모니터링 기능이 대폭 강화되었습니다. 주요 변경 사항과 작동 방식은 다음과 같습니다.

## 1. 신규 기능: 시스템 건전성 모니터링 (System Health)
실시간 분석 데이터를 바탕으로 전체 시스템의 상태를 한눈에 파악할 수 있는 전용 페이지가 추가되었습니다.

### 주요 지표 (Metrics)
*   **System Status**: 알림 발생 여부에 따라 `HEALTHY` (정상) 또는 `CRITICAL/WARNING` (주의) 상태 표시
*   **Low Confidence Ratio**: 분석 결과 중 신뢰도가 낮은 결과의 비율 제공
*   **Defects per Item**: 제품 한 개당 평균 결함 발견 수치 시각화
*   **Confidence Distribution**: 결함의 신뢰도 수준을 4단계(High, Medium, Low, Very Low)로 나누어 도넛 차트로 표시
*   **Defect Type Distribution**: 발견된 결함 유형별 총 개수를 가로 막대 차트로 표시

## 2. 세션 관리 및 데이터 유지 (Persistence)
*   **세션 유지**: 사용자가 선택한 세션 ID를 `sessionStorage`에 저장하여 페이지를 새로고침하거나 다른 탭(Dashboard <-> System Health)으로 이동해도 선택한 세션이 그대로 유지됩니다.
*   **세션 선택기 동기화**: 모든 페이지의 상단 네비게이션 바에서 동일한 세션 선택기를 제공하며, 변경 시 모든 차트와 지표가 즉시 업데이트됩니다.

## 3. 백엔드 연동 및 작동 방식
시스템은 백엔드 API로부터 데이터를 실시간으로 가져와 동적으로 화면을 갱신합니다.

### 주요 API 엔드포인트
*   **`/monitoring/health`**: 시스템 상태, 통합 통계, 신뢰도 분포, 결함 유형 분포 데이터 제공
*   **`/monitoring/alerts`**: 실시간 시스템 경고 및 조치 권장 사항 제공
*   **`/sessions`**: 세션 목록 및 시작/종료 시간 정보 제공
*   **`/latest`**: 최신 결함 이미지 및 탐지 결과(BBox 포함) 제공

### 데이터 렌더링 메커니즘
1.  **Polling**: `api-client.js`가 1~2초 주기로 백엔드에 새로운 데이터를 요청합니다.
2.  **Chart.js**: 가져온 데이터를 `Chart.js` 라이브러리를 통해 동적 차트로 렌더링합니다.
3.  **Proxy Server**: `gulpfile.js`에 설정된 프록시 미들웨어를 통해 브라우저의 `/api` 요청을 실제 백엔드 서버(`127.0.0.1:8000`)로 안전하게 전달합니다.

## 4. UI/UX 개선 사항
*   **사이드바 정리**: 불필요한 'Table List' 탭을 제거하고 꼭 필요한 메뉴(Dashboard, Defect Logs, System Health)로 간소화했습니다.
*   **로고 링크**: 좌측 상단의 로고와 'Merby' 텍스트에 메인 대시보드로 이동하는 링크를 추가하여 접근성을 높였습니다.
*   **알림 배너**: 중복되는 수치 표시를 제거하여 경고 메시지의 가독성을 개선했습니다.

---
**실행 방법**:
1. 백엔드: `uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
2. 프론트엔드: `conda run -n dashboard-env npm start`
