# API Key 인증 구현 가이드

## 📋 개요
Edge Device(Jetson)와 Backend Server(FastAPI) 간의 통신에 API Key 인증을 추가하여 보안을 강화합니다.

---

## ✅ 완료된 작업 (코드 수정)

### 1. Backend (FastAPI) 수정
- **`serving/api/config/settings.py`**
  - `dotenv` 로딩 추가
  - `EDGE_API_KEY` 환경 변수 설정 추가

- **`serving/api/utils/auth.py`** (신규 생성)
  - API Key 검증 로직 구현
  - `verify_api_key()` 함수: HTTP 헤더의 `X-API-KEY`를 검증

- **`serving/api/routers/detect.py`**
  - `/detect` 엔드포인트에 인증 의존성 추가
  - `dependencies=[Depends(verify_api_key)]` 적용

### 2. Edge (Jetson) 수정
- **`serving/edge/inference_worker.py`**
  - `dotenv` 로딩 추가
  - API 요청 시 `X-API-KEY` 헤더 추가
  - 환경 변수에서 API Key 읽어오도록 수정

---

## 🔧 서버 설정 작업 (직접 수행 필요)

### 1. Backend Server (EC2: 3.35.182.98)

#### 작업 위치
```bash
/home/ubuntu/pro-cv-finalproject-cv-01/serving/api/
```

#### 수행 작업
1. SSH로 EC2 서버 접속
2. `.env` 파일 생성 또는 수정
   ```bash
   cd /home/ubuntu/pro-cv-finalproject-cv-01/serving/api/
   nano .env
   ```

3. 다음 내용 추가 (기존 AWS, Slack 설정 유지)
   ```bash
   # Edge Device Authentication
   EDGE_API_KEY=pcb_detector_secret_2026_your_custom_key #예시
   ```

4. 파일 저장 후 FastAPI 서버 재시작
   ```bash
   # 서버 재시작 명령 (실제 명령은 배포 방식에 따라 다를 수 있음)
   sudo systemctl restart pcb-api
   # 또는
   pkill -f uvicorn && uvicorn main:app --host 0.0.0.0 --port 8080
   ```

#### 주의사항
- `.env` 파일은 절대 Git에 커밋하지 말 것 (이미 `.gitignore`에 포함됨)
- API Key는 충분히 복잡하게 설정 (최소 20자 이상 권장)

---

### 2. Edge Device (Jetson)

#### 작업 위치
```bash
# Jetson 내부 프로젝트 디렉토리 (실제 경로는 다를 수 있음)
/home/nvidia/project/serving/edge/
```

#### 수행 작업
1. SSH로 Jetson 접속
2. `.env` 파일 생성
   ```bash
   cd /home/nvidia/project/serving/edge/
   nano .env
   ```

3. **Backend와 동일한 API Key** 입력
   ```bash
   # Edge Device Authentication (Backend와 동일한 값)
   EDGE_API_KEY=pcb_detector_secret_2026_your_custom_key
   ```

4. 파일 저장 후 Edge 프로그램 재시작
   ```bash
   # 실행 중인 프로세스 종료 후 재시작
   pkill -f main.py
   python main.py --input rtsp://... --api-url http://3.35.182.98:8080/detect/
   ```

#### 주의사항
- Backend와 **완전히 동일한 API Key**를 사용해야 함
- Jetson이 여러 대인 경우, 모든 장비에 동일하게 설정
- (선택) 장비별로 다른 키를 사용하려면 Backend에서 키 목록 관리 필요

---

## 🧪 테스트 방법

### 1. 성공 케이스 (올바른 API Key)
```bash
curl -X POST http://3.35.182.98:8080/detect \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: pcb_detector_secret_2026_your_custom_key" \
  -d '{
    "timestamp": "2026-02-02T13:00:00",
    "image_id": "test_001",
    "detections": []
  }'
```
**예상 결과**: `200 OK` 또는 `201 Created`

### 2. 실패 케이스 (API Key 없음)
```bash
curl -X POST http://3.35.182.98:8080/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-02-02T13:00:00",
    "image_id": "test_002",
    "detections": []
  }'
```
**예상 결과**: `403 Forbidden` - "API Key가 필요합니다"

### 3. 실패 케이스 (잘못된 API Key)
```bash
curl -X POST http://3.35.182.98:8080/detect \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: wrong_key_123" \
  -d '{
    "timestamp": "2026-02-02T13:00:00",
    "image_id": "test_003",
    "detections": []
  }'
```
**예상 결과**: `403 Forbidden` - "유효하지 않은 API Key입니다"

---

## 📊 동작 원리

### 인증 흐름
```
1. Jetson (Edge)
   ↓ HTTP POST /detect
   ↓ Header: X-API-KEY: pcb_detector_secret_2026_your_custom_key
   
2. FastAPI (Backend)
   ↓ verify_api_key() 함수 실행
   ↓ 헤더의 키 vs 환경변수의 키 비교
   
3-A. 일치 ✅
   → 원래 함수 실행 (DB 저장, S3 업로드)
   
3-B. 불일치 ❌
   → 403 Forbidden 즉시 반환
   → DB/S3 접근 차단
```

### 보안 효과
- ✅ 허가되지 않은 디바이스의 데이터 주입 차단
- ✅ 데이터 오염(Data Poisoning) 방지
- ✅ API 엔드포인트 무단 접근 차단
- ✅ 로그를 통한 접근 추적 가능

---

## 🔒 보안 권장사항

### 1. API Key 관리
- 주기적으로 키 변경 (예: 3개월마다)
- 키 유출 시 즉시 변경 및 로그 확인
- 키는 최소 20자 이상, 영문+숫자+특수문자 조합

### 2. 환경 변수 보호
- `.env` 파일 권한 설정: `chmod 600 .env`
- 절대 Git, 로그, 슬랙 등에 노출 금지
- 서버 백업 시 `.env` 제외 확인

### 3. 추가 보안 강화 (선택)
- Rate Limiting: 동일 IP에서 과도한 요청 차단
- IP Whitelist: 특정 IP에서만 접근 허용
- HTTPS 적용: 통신 구간 암호화
- API Key 만료 시간 설정

---

## 📝 체크리스트

### Backend (EC2) 설정
- [ ] `/home/ubuntu/.../serving/api/.env` 파일 생성
- [ ] `EDGE_API_KEY` 값 설정 (20자 이상)
- [ ] FastAPI 서버 재시작
- [ ] `/docs` 페이지에서 인증 필드 확인

### Edge (Jetson) 설정
- [ ] Jetson 내부 `.env` 파일 생성
- [ ] Backend와 동일한 `EDGE_API_KEY` 값 입력
- [ ] Edge 프로그램 재시작
- [ ] 로그에서 "전송 성공" 메시지 확인

### 테스트
- [ ] curl로 성공 케이스 테스트
- [ ] curl로 실패 케이스 테스트 (키 없음)
- [ ] curl로 실패 케이스 테스트 (잘못된 키)
- [ ] 실제 Jetson에서 데이터 전송 확인

---

## 🆘 트러블슈팅

### 문제: "ModuleNotFoundError: No module named 'dotenv'"
**원인**: python-dotenv 패키지 미설치  
**해결**: 
```bash
pip install python-dotenv
# 또는
uv pip install python-dotenv
```

### 문제: Jetson에서 403 Forbidden 계속 발생
**원인**: API Key 불일치 또는 헤더 누락  
**해결**:
1. Jetson의 `.env` 파일 확인
2. Backend의 `.env` 파일과 값 비교
3. `inference_worker.py` 로그 확인 (헤더가 제대로 전송되는지)

### 문제: Backend 서버가 시작되지 않음
**원인**: settings.py에서 환경 변수 로딩 실패  
**해결**:
1. `.env` 파일이 올바른 위치에 있는지 확인
2. `load_dotenv()` 호출 위치 확인
3. 서버 로그에서 에러 메시지 확인

---

## 📅 작업 이력
- 2026-02-02: API Key 인증 기능 구현 완료 (코드 수정)
- 다음 단계: EC2 및 Jetson 서버 설정 작업 필요
