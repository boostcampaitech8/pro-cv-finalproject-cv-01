"""
Health API 테스트 스크립트
다양한 임계값을 넘는 데이터를 생성하여 알림 시스템 검증
"""
import requests
import random
from datetime import datetime

BASE_URL = "http://localhost:8000"


def create_test_session(name: str, model_name: str = None):
    """테스트 세션 생성

    model_name을 보내면 서버가 해당 정보를 기록한다.
    """
    print(f"\n{'='*60}")
    print(f"테스트 세션: {name} (model={model_name})")
    print('='*60)

    payload = {}
    if model_name is not None:
        payload["model_name"] = model_name

    resp = requests.post(f"{BASE_URL}/sessions/", json=payload if payload else None)
    data = resp.json()
    session_id = data.get("id")
    print(f"세션 ID: {session_id}, model stored: {data.get('model_name')}")
    return session_id


def send_inspection(session_id: int, image_id: str, defects: list):
    """검사 결과 전송"""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "image_id": image_id,
        "detections": defects,
        "session_id": session_id
    }

    resp = requests.post(f"{BASE_URL}/detect/", json=payload)
    return resp.json()


def check_health(session_id: int):
    """Health API 호출 및 결과 출력"""
    resp = requests.get(f"{BASE_URL}/monitoring/health?session_id={session_id}")
    data = resp.json()

    print(f"\n📊 통계:")
    print(f"  - 총 검사: {data['total_inspections']}개")
    print(f"  - 정상: {data['normal_count']}개")
    print(f"  - 불량: {data['defect_count']}개")
    print(f"  - 불량률: {data['defect_rate']:.1f}%")
    print(f"  - 총 결함: {data['total_defects']}개")
    print(f"  - 불량당 평균 결함: {data['avg_defects_per_item']:.2f}개")

    if data['defect_confidence_stats']:
        stats = data['defect_confidence_stats']
        print(f"\n🎯 신뢰도 통계:")
        print(f"  - 평균: {stats['avg_confidence']:.3f}")
        print(f"  - 최소: {stats['min_confidence']:.3f}")
        print(f"  - 최대: {stats['max_confidence']:.3f}")
        dist = stats['distribution']
        print(f"  - 분포: 높음({dist['high']}) / 중간({dist['medium']}) / 낮음({dist['low']}) / 매우낮음({dist['very_low']})")

    if data['defect_type_stats']:
        print(f"\n🔍 결함 타입:")
        for dtype in data['defect_type_stats']:
            print(f"  - {dtype['defect_type']}: {dtype['count']}개 (평균 신뢰도: {dtype['avg_confidence']:.3f})")

    print(f"\n⚠️  알림 ({len(data['alerts'])}개):")
    if data['alerts']:
        for alert in data['alerts']:
            emoji = "🔴" if alert['level'] == "critical" else "🟡"
            print(f"  {emoji} [{alert['level'].upper()}] {alert['message']}")
            print(f"     현재값: {alert['value']:.2f} / 임계값: {alert['threshold']:.2f}")
            print(f"     조치: {alert['action']}")
    else:
        print("  ✅ 정상 상태 (알림 없음)")

    print(f"\n🏥 시스템 상태: {data['status'].upper()}")

    return data


def test_scenario_1_healthy():
    """시나리오 1: 정상 상태 (알림 없음)"""
    session_id = create_test_session("시나리오 1: 정상 상태", model_name="yolov11m_v0")

    # 확인: /sessions 리스트에 모델명이 저장되어 있는지
    sessions_resp = requests.get(f"{BASE_URL}/sessions/")
    sessions_data = sessions_resp.json()["sessions"]
    matching = [s for s in sessions_data if s["id"] == session_id]
    assert matching and matching[0].get("model_name") == "yolov11m_v0"

    # 불량률 5% (50개 중 2~3개 불량)
    for i in range(50):
        is_defect = i < 2

        defects = []
        if is_defect:
            defects.append({
                "defect_type": "scratch",
                "confidence": random.uniform(0.92, 0.98),
                "bbox": [10, 20, 100, 120]
            })

        send_inspection(session_id, f"test_s1_{i:04d}", defects)

    check_health(session_id)


def test_scenario_2_high_defect_rate():
    """시나리오 2: 높은 불량률 (warning 또는 critical)"""
    session_id = create_test_session("시나리오 2: 높은 불량률", model_name="yolov11m_v0")

    sessions_resp = requests.get(f"{BASE_URL}/sessions/")
    sessions_data = sessions_resp.json()["sessions"]
    matching = [s for s in sessions_data if s["id"] == session_id]
    assert matching and matching[0].get("model_name") == "yolov11m_v0"

    # 불량률 25% (40개 중 10개 불량)
    for i in range(40):
        is_defect = i < 10

        defects = []
        if is_defect:
            defects.append({
                "defect_type": random.choice(["scratch", "hole", "crack"]),
                "confidence": random.uniform(0.88, 0.96),
                "bbox": [10, 20, 100, 120]
            })

        send_inspection(session_id, f"test_s2_{i:04d}", defects)

    check_health(session_id)


def test_scenario_3_low_confidence():
    """시나리오 3: 낮은 신뢰도"""
    session_id = create_test_session("시나리오 3: 낮은 신뢰도")

    # 불량률은 낮지만 신뢰도가 낮음
    for i in range(30):
        is_defect = i < 3

        defects = []
        if is_defect:
            # 낮은 신뢰도
            defects.append({
                "defect_type": "scratch",
                "confidence": random.uniform(0.68, 0.74),  # very_low 또는 low
                "bbox": [10, 20, 100, 120]
            })

        send_inspection(session_id, f"test_s3_{i:04d}", defects)

    check_health(session_id)


def test_scenario_4_multiple_defects_per_item():
    """시나리오 4: PCB당 다중 결함"""
    session_id = create_test_session("시나리오 4: PCB당 다중 결함")

    # 불량률 12%, PCB당 평균 3.5개 결함
    for i in range(50):
        is_defect = i < 6

        defects = []
        if is_defect:
            # 한 PCB에 3~4개 결함
            num_defects = random.randint(3, 4)
            for _ in range(num_defects):
                defects.append({
                    "defect_type": random.choice(["scratch", "hole", "crack", "short"]),
                    "confidence": random.uniform(0.85, 0.95),
                    "bbox": [
                        random.randint(0, 400),
                        random.randint(0, 400),
                        random.randint(50, 500),
                        random.randint(50, 500)
                    ]
                })

        send_inspection(session_id, f"test_s4_{i:04d}", defects)

    check_health(session_id)


def test_scenario_5_multiple_alerts():
    """시나리오 5: 여러 알림 동시 발생"""
    session_id = create_test_session("시나리오 5: 복합 문제 (여러 알림)")

    # 높은 불량률 + 낮은 신뢰도 + 다중 결함
    for i in range(30):
        is_defect = i < 8  # 26% 불량률

        defects = []
        if is_defect:
            # PCB당 2~4개 결함, 낮은 신뢰도 포함
            num_defects = random.randint(2, 4)
            for j in range(num_defects):
                # 일부는 낮은 신뢰도
                if j == 0 and random.random() < 0.5:
                    conf = random.uniform(0.65, 0.75)
                else:
                    conf = random.uniform(0.80, 0.92)

                defects.append({
                    "defect_type": random.choice(["scratch", "hole", "crack", "short", "open_circuit"]),
                    "confidence": conf,
                    "bbox": [
                        random.randint(0, 400),
                        random.randint(0, 400),
                        random.randint(50, 500),
                        random.randint(50, 500)
                    ]
                })

        send_inspection(session_id, f"test_s5_{i:04d}", defects)

    check_health(session_id)


if __name__ == "__main__":
    print("Health Monitoring API 테스트 시작")
    print("="*60)

    # 각 시나리오 실행
    test_scenario_1_healthy()
    test_scenario_2_high_defect_rate()
    test_scenario_3_low_confidence()
    test_scenario_4_multiple_defects_per_item()
    test_scenario_5_multiple_alerts()

    print("\n" + "="*60)
    print("✅ 모든 테스트 완료")
    print("="*60)
