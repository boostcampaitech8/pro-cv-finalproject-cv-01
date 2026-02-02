"""
POST /feedback API 테스트 스크립트

테스트 시나리오:
1. False positive 피드백 생성
2. Label correction 피드백 생성
3. 존재하지 않는 log_id → 404 에러
4. label_correction + correct_label 누락 → 422 에러
5. correct_label 허용값 외 → 422 에러
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(test_name, success, response=None, error=None):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if response:
        print(f"  Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    if error:
        print(f"  Error: {error}")
    print()


def setup_test_data():
    """테스트용 검사 데이터 생성"""
    print_section("테스트 데이터 준비")

    # 정상 PCB 1개
    normal_data = {
        "timestamp": datetime.now().isoformat(),
        "image_id": "TEST_PCB_001",
        "detections": []
    }

    # 불량 PCB 1개 (scratch)
    defect_data = {
        "timestamp": datetime.now().isoformat(),
        "image_id": "TEST_PCB_002",
        "detections": [
            {
                "defect_type": "scratch",
                "confidence": 0.95,
                "bbox": [10, 20, 100, 120]
            }
        ]
    }

    # 데이터 전송
    r1 = requests.post(f"{BASE_URL}/detect", json=normal_data)
    r2 = requests.post(f"{BASE_URL}/detect", json=defect_data)

    if r1.status_code == 200 and r2.status_code == 200:
        normal_log_id = r1.json()["id"]
        defect_log_id = r2.json()["id"]
        print(f"✅ 테스트 데이터 생성 완료")
        print(f"  정상 PCB log_id: {normal_log_id}")
        print(f"  불량 PCB log_id: {defect_log_id}")
        return normal_log_id, defect_log_id
    else:
        print("❌ 테스트 데이터 생성 실패")
        return None, None


def test_false_positive(log_id):
    """테스트 1: False Positive 피드백"""
    print_section("테스트 1: False Positive 피드백")

    payload = {
        "log_id": log_id,
        "feedback_type": "false_positive",
        "comment": "정상 PCB인데 먼지로 인해 오탐됨",
        "created_by": "manager_kim"
    }

    try:
        response = requests.post(f"{BASE_URL}/feedback/", json=payload)
        success = response.status_code == 201
        print_result("False Positive 피드백 생성", success, response)
        return success
    except Exception as e:
        print_result("False Positive 피드백 생성", False, error=str(e))
        return False


def test_label_correction(log_id):
    """테스트 2: Label Correction 피드백"""
    print_section("테스트 2: Label Correction 피드백")

    payload = {
        "log_id": log_id,
        "feedback_type": "label_correction",
        "correct_label": "hole",
        "comment": "scratch가 아니라 hole입니다"
    }

    try:
        response = requests.post(f"{BASE_URL}/feedback/", json=payload)
        success = response.status_code == 201
        print_result("Label Correction 피드백 생성", success, response)
        return success
    except Exception as e:
        print_result("Label Correction 피드백 생성", False, error=str(e))
        return False


def test_nonexistent_log_id():
    """테스트 3: 존재하지 않는 log_id → 404 에러"""
    print_section("테스트 3: 존재하지 않는 log_id (예상: 404)")

    payload = {
        "log_id": 99999,
        "feedback_type": "false_positive"
    }

    try:
        response = requests.post(f"{BASE_URL}/feedback/", json=payload)
        success = response.status_code == 404
        print_result("존재하지 않는 log_id 에러 처리", success, response)
        return success
    except Exception as e:
        print_result("존재하지 않는 log_id 에러 처리", False, error=str(e))
        return False


def test_missing_correct_label(log_id):
    """테스트 4: label_correction + correct_label 누락 → 422 에러"""
    print_section("테스트 4: label_correction + correct_label 누락 (예상: 422)")

    payload = {
        "log_id": log_id,
        "feedback_type": "label_correction"
    }

    try:
        response = requests.post(f"{BASE_URL}/feedback/", json=payload)
        success = response.status_code == 422
        print_result("correct_label 누락 검증", success, response)
        return success
    except Exception as e:
        print_result("correct_label 누락 검증", False, error=str(e))
        return False


def test_invalid_correct_label(log_id):
    """테스트 5: correct_label 허용값 외 → 422 에러"""
    print_section("테스트 5: correct_label 허용값 외 (예상: 422)")

    payload = {
        "log_id": log_id,
        "feedback_type": "label_correction",
        "correct_label": "unknown_defect"
    }

    try:
        response = requests.post(f"{BASE_URL}/feedback/", json=payload)
        success = response.status_code == 422
        print_result("correct_label 허용값 검증", success, response)
        return success
    except Exception as e:
        print_result("correct_label 허용값 검증", False, error=str(e))
        return False


def test_false_negative(log_id):
    """테스트 6: False Negative 피드백"""
    print_section("테스트 6: False Negative 피드백")

    payload = {
        "log_id": log_id,
        "feedback_type": "false_negative",
        "comment": "불량인데 정상으로 통과됨",
        "created_by": "manager_lee"
    }

    try:
        response = requests.post(f"{BASE_URL}/feedback/", json=payload)
        success = response.status_code == 201
        print_result("False Negative 피드백 생성", success, response)
        return success
    except Exception as e:
        print_result("False Negative 피드백 생성", False, error=str(e))
        return False


def main():
    print_section("POST /feedback API 통합 테스트")

    # 서버 연결 확인
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ 서버 연결 성공: {BASE_URL}")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        print("서버를 먼저 시작하세요: uv run uvicorn main:app --reload --port 8000")
        return

    # 테스트 데이터 준비
    normal_log_id, defect_log_id = setup_test_data()
    if not normal_log_id or not defect_log_id:
        print("❌ 테스트 데이터 생성 실패로 테스트 중단")
        return

    # 테스트 실행
    results = []
    results.append(("False Positive", test_false_positive(defect_log_id)))
    results.append(("Label Correction", test_label_correction(defect_log_id)))
    results.append(("존재하지 않는 log_id", test_nonexistent_log_id()))
    results.append(("correct_label 누락", test_missing_correct_label(defect_log_id)))
    results.append(("correct_label 허용값 외", test_invalid_correct_label(defect_log_id)))
    results.append(("False Negative", test_false_negative(normal_log_id)))

    # 결과 요약
    print_section("테스트 결과 요약")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 모든 테스트 통과!")
    else:
        print(f"\n⚠️  {total - passed}개 테스트 실패")


if __name__ == "__main__":
    main()
