# 프론트엔드 (Dashboard)

## 개요

PCB 결함 탐지 시스템의 실시간 모니터링 대시보드입니다.

**접속 URL:** http://3.35.182.98/

## 기술 스택

| 구분 | 기술 | 설명 |
|------|------|------|
| 템플릿 | Paper Dashboard 2 | Bootstrap 기반 무료 대시보드 템플릿 |
| 프레임워크 | Bootstrap 4 | 반응형 UI 프레임워크 |
| 언어 | HTML/CSS/JavaScript | Vanilla JS (프레임워크 없음) |
| 서버 | nginx | 정적 파일 서빙 + API 프록시 |

## 프로젝트 구조

```
dashboard/
├── index.html              # 랜딩 페이지
├── dashboard.html          # 메인 대시보드
├── health.html             # 헬스 모니터링 페이지
├── assets/
│   ├── css/                # 스타일시트
│   ├── js/                 # JavaScript
│   ├── img/                # 이미지
│   └── fonts/              # 폰트
└── examples/               # 템플릿 예제
```

## 주요 페이지

### 1. 메인 대시보드 (dashboard.html)

**기능:**
- 실시간 통계 표시 (총 검사 수, 불량률, 정상/불량 개수)
- 최근 검사 이력 테이블 (최신 10건)
- 결함 타입별 집계 차트
- 1초마다 자동 업데이트 (폴링)

**API 연동:**
```javascript
// 통계 조회
fetch('/api/stats')
  .then(res => res.json())
  .then(data => {
    // 화면 업데이트
  });

// 최근 검사 이력
fetch('/api/latest?limit=10')
  .then(res => res.json())
  .then(data => {
    // 테이블 업데이트
  });

// 결함 타입별 집계
fetch('/api/defects')
  .then(res => res.json())
  .then(data => {
    // 차트 업데이트
  });
```

**주요 컴포넌트:**
- **통계 카드**: 총 검사 수, 불량률, 정상/불량 개수
- **검사 이력 테이블**: 시각, 이미지 ID, 결과, 결함 타입, 신뢰도
- **결함 타입 차트**: Chart.js 바 차트

### 2. 헬스 모니터링 페이지 (health.html)

**기능:**
- 시스템 건강 상태 표시 (healthy/warning/critical)
- 알림 목록 (불량률, 신뢰도, 저신뢰도 비율, PCB당 평균 결함)
- 신뢰도 분포 차트
- 결함 타입별 통계
- 1초마다 자동 업데이트 (폴링)

**API 연동:**
```javascript
// 헬스 모니터링
fetch('/api/monitoring/health')
  .then(res => res.json())
  .then(data => {
    // 상태 표시, 알림 목록, 차트 업데이트
  });
```

**주요 컴포넌트:**
- **상태 배지**: healthy (초록), warning (노랑), critical (빨강)
- **알림 목록**: Critical/Warning 알림 분리 표시
- **신뢰도 분포 차트**: high/medium/low/very_low 구간별 도넛 차트
- **결함 타입 통계 테이블**: 타입별 발생 횟수, 평균 신뢰도

## API 연동 방법

### Nginx 프록시 설정

프론트엔드는 nginx를 통해 백엔드 API에 접근합니다. `/api/*` 경로가 백엔드 서버로 프록시됩니다.

```nginx
# /etc/nginx/sites-available/default
server {
    listen 80;
    server_name 3.35.182.98;

    # 정적 파일 (프론트엔드)
    location / {
        root /home/ubuntu/app/dashboard;
        index index.html;
        try_files $uri $uri/ =404;
    }

    # API 프록시 (백엔드)
    location /api/ {
        proxy_pass http://localhost:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**예시:**
- 브라우저 요청: `http://3.35.182.98/api/stats`
- nginx 프록시: `http://localhost:8080/stats`

### API 엔드포인트

프론트엔드에서 사용하는 주요 API:

| 엔드포인트 | 메서드 | 설명 | 폴링 주기 |
|-----------|--------|------|-----------|
| `/api/stats` | GET | 통계 정보 | 1초 |
| `/api/latest` | GET | 최근 검사 이력 | 1초 |
| `/api/defects` | GET | 결함 타입별 집계 | 1초 |
| `/api/monitoring/health` | GET | 헬스 모니터링 | 1초 |
| `/api/monitoring/alerts` | GET | 알림 목록 (경량) | 1초 |

상세 API 명세는 [backend/api.md](backend/api.md) 참조.

## 실행 방법

### 로컬 개발

```bash
# Python HTTP 서버로 간단히 실행
cd dashboard
python3 -m http.server 8000
```

브라우저: http://localhost:8000/

**주의:** 로컬에서는 API 프록시가 없으므로, 백엔드 API 주소를 직접 입력해야 합니다. (CORS 설정 필요)

### 운영 배포 (nginx)

**파일 배포:**
```bash
# EC2 서버에서
cd /home/ubuntu/app
git pull origin dev
```

**nginx 재시작:**
```bash
sudo systemctl reload nginx
```

**헬스체크:**
```bash
curl http://3.35.182.98/
```

## 배포 정보

| 구분 | 내용 |
|------|------|
| 서버 | EC2 (3.35.182.98) |
| 포트 | 80 (HTTP) |
| 배포 경로 | /home/ubuntu/app/dashboard/ |
| 서버 소프트웨어 | nginx |
| CI/CD | GitHub Actions (dev 브랜치 push 시 자동 배포) |

## 커스터마이징

### 폴링 주기 변경

```javascript
// dashboard.html, health.html의 setInterval 수정
setInterval(fetchStats, 1000);  // 1000ms = 1초
```

### 차트 색상 변경

```javascript
// Chart.js 옵션 수정
backgroundColor: [
  'rgba(255, 99, 132, 0.2)',   // 빨강
  'rgba(54, 162, 235, 0.2)',   // 파랑
  'rgba(255, 206, 86, 0.2)',   // 노랑
  'rgba(75, 192, 192, 0.2)'    // 초록
]
```

### 테이블 항목 수 변경

```javascript
// latest API 호출 시 limit 파라미터 변경
fetch('/api/latest?limit=20')  // 10 → 20
```

## 브라우저 지원

- Chrome (권장)
- Firefox
- Safari
- Edge

**IE 11 이하는 지원하지 않습니다.**

## 참고 자료

- [Paper Dashboard 2 문서](https://demos.creative-tim.com/paper-dashboard-2/docs/1.0/getting-started/introduction.html)
- [Chart.js 문서](https://www.chartjs.org/docs/latest/)
- [Bootstrap 4 문서](https://getbootstrap.com/docs/4.6/getting-started/introduction/)
