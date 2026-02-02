# PCB Defect Detection Dashboard

PCB 결함 탐지 시스템의 실시간 모니터링을 위한 대시보드 프로젝트입니다. Paper Dashboard 2 템플릿을 기반으로 하며, FastAPI 백엔드와 연결되어 실시간 데이터를 시각화합니다.

## 📂 폴더 및 파일 구조

### 1. Root Files
- `index.html`: 대시보드 접속 시 메인 페이지(`examples/dashboard.html`)로 자동 리다이렉트합니다.
- `package.json` & `gulpfile.js`: 개발 서버 및 빌드 설정 파일입니다.

### 2. 페이지 구성 (`examples/`)

#### [실제 사용 중인 핵심 페이지]
- `dashboard.html`: **메인 화면**. 실시간 통계 카드, 결함 추이 차트, 유형 분포, 신뢰도 차트를 포함합니다. 백엔드 API와 실시간으로 연동되는 핵심 파일입니다.
- `icons.html`: **결함 로그(Defect Logs) 확인 페이지**. 기존 템플릿의 아이콘 페이지를 커스텀하여, 발생한 결함들의 상세 이력을 확인하는 용도로 사용하고 있습니다.

#### [참고용 템플릿 샘플 (현재 프로젝트와 무관)]
다음 파일들은 Paper Dashboard 템플릿에 포함된 샘플 페이지들로, 현재 프로젝트의 기능이나 데이터와는 연결되어 있지 않습니다. 향후 UI 확장 시 레이아웃 참고용으로만 활용됩니다.
- `tables.html`: 단순 테이블 디자인 샘플 (Name, Country 등 가상 데이터)
- `map.html`, `typography.html`, `user.html`, `notifications.html`: 기타 UI 구성 요소 샘플

### 3. 자산 (`assets/`)
- `js/api-client.js`: **연동 핵심**. 백엔드 API 서버와 통신하며 데이터를 가져오고 화면을 갱신하는 모든 로직이 들어있습니다.
- `css/` & `scss/`: 대시보드 스타일 시트입니다.
- `js/plugins/`: Chart.js, jQuery 등 외부 라이브러리가 포함되어 있습니다.

## 🖥 웹 구성 요소 (dashboard.html)

대시보드 화면은 다음과 같은 실시간 요소들로 구성되어 있습니다.

1.  **통계 카드 (Stats Cards)**
    *   **Total Count**: 전체 검사 수
    *   **Normal Count**: 정상 제품 수
    *   **Defect Count**: 결함 발견 수
    *   **Model Used**: 현재 사용 중인 탐지 모델 정보
2.  **Defect Trends (Line Chart)**: 시간대별 결함 발생 추이를 보여줍니다.
3.  **Defect Types (Pie Chart)**: 발견된 결함 종류별 분포(Short, Open, Spur 등)를 표시합니다.
4.  **Recent Defect Confidence (Line Chart)**: 최근 발견된 결함들의 탐지 신뢰도 점수를 시각화합니다.
5.  **Session Selector**: 우측 상단 드롭다운을 통해 특정 작업 세션별로 데이터를 필터링할 수 있습니다.

## 🔗 백엔드 연결 정보 (api-client.js)

대시보드는 백엔드 API 서버와 `fetch` 통신을 통해 데이터를 주고받습니다.

-   **연결 설정**: `const API_BASE_URL = "/api";` (코드 3라인)
-   **데이터 업데이트**: `DashboardUpdater.startPolling()` 함수가 1초마다 백엔드에 새로운 데이터를 요청합니다.

### 주요 API 호출 및 연동 코드
| 기능 | 호출 함수 (api-client.js) | 백엔드 엔드포인트 | 비고 |
| :--- | :--- | :--- | :--- |
| 세션 목록 조회 | `getSessions()` | `GET /api/sessions/` | 세션 드롭다운 구성 |
| 실시간 통계 | `getStats(sessionId)` | `GET /api/stats` | 상단 카드 및 트렌드 차트용 |
| 결함 유형 집계 | `getDefects(sessionId)` | `GET /api/defects` | 원형 차트용 |
| 최근 검사 로그 | `getLatest(limit, sessionId)` | `GET /api/latest` | 신뢰도 차트용 |

## 🚀 실행 방법

1.  **백엔드 서버 실행**: `ss/serving/api` 폴더에서 FastAPI 서버가 실행 중이어야 합니다.
2.  **대시보드 실행**:
    ```bash
    npm install
    npm run start
    ```
    이 명령어를 실행하면 `gulpfile.js` 설정에 따라 로컬 서버가 실행되며 브라우저가 열립니다.
