"""
피드백 통계 API 테스트

테스트 항목:
1. GET /feedback/stats - 피드백 통계 조회
2. 응답 스키마 검증
3. 결함 타입별 집계 검증
"""

import requests
import sys

BASE_URL = "http://localhost:8000"


def print_section(title):
    """섹션 제목 출력"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_feedback_stats():
    """피드백 통계 조회 테스트"""
    print_section("GET /feedback/stats 테스트")

    response = requests.get(f"{BASE_URL}/feedback/stats")

    print(f"Status Code: {response.status_code}")

    if response.status_code != 200:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        print(f"Response: {response.text}")
        return False

    stats = response.json()

    # 스키마 검증
    required_fields = [
        "total_feedback",
        "by_type",
        "by_defect_type",
        "recent_feedback_count",
        "period_description"
    ]

    for field in required_fields:
        if field not in stats:
            print(f"❌ FAIL: Missing field '{field}'")
            return False

    # by_type 검증
    by_type = stats["by_type"]
    type_fields = ["false_positive", "false_negative", "label_correction"]
    for field in type_fields:
        if field not in by_type:
            print(f"❌ FAIL: Missing by_type field '{field}'")
            return False

    print(f"✅ PASS GET /feedback/stats")
    print(f"\n📊 통계 요약:")
    print(f"   총 피드백: {stats['total_feedback']}개")
    print(f"   오탐 (FP): {by_type['false_positive']}개")
    print(f"   미탐 (FN): {by_type['false_negative']}개")
    print(f"   라벨 수정: {by_type['label_correction']}개")
    print(f"   최근 24시간: {stats['recent_feedback_count']}개")
    print(f"   집계 기간: {stats['period_description']}")

    # 결함 타입별 출력
    if stats["by_defect_type"]:
        print(f"\n   결함 타입별 집계:")
        for item in stats["by_defect_type"]:
            total = (item["false_positive"] +
                    item["false_negative"] +
                    item["label_correction"])
            print(f"   - {item['defect_type']}: "
                  f"FP={item['false_positive']}, "
                  f"FN={item['false_negative']}, "
                  f"LC={item['label_correction']} "
                  f"(합계: {total})")
    else:
        print(f"\n   결함 타입별 집계: 없음 (불량 피드백 없음)")

    return True


def test_empty_stats():
    """피드백이 없을 때 통계 조회 테스트"""
    print_section("빈 통계 테스트 (피드백 없을 때)")

    response = requests.get(f"{BASE_URL}/feedback/stats")

    if response.status_code != 200:
        print(f"❌ FAIL: Expected 200, got {response.status_code}")
        return False

    stats = response.json()

    # 피드백이 있으면 이 테스트는 스킵
    if stats["total_feedback"] > 0:
        print(f"⏭️  SKIP: 피드백 데이터 존재 ({stats['total_feedback']}개)")
        return True

    # 빈 통계 검증
    if stats["total_feedback"] != 0:
        print(f"❌ FAIL: Expected total_feedback=0")
        return False

    if stats["by_type"]["false_positive"] != 0:
        print(f"❌ FAIL: Expected false_positive=0")
        return False

    if len(stats["by_defect_type"]) != 0:
        print(f"❌ FAIL: Expected empty by_defect_type")
        return False

    print(f"✅ PASS 빈 통계 테스트")
    return True


def main():
    """메인 테스트 실행"""
    print("\n" + "="*60)
    print("  피드백 통계 API 테스트")
    print("="*60)
    print(f"API Base URL: {BASE_URL}")

    # 서버 상태 확인
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=2)
        print(f"✅ 서버 상태: OK (Status {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ 서버 연결 실패: {e}")
        print(f"\n서버를 먼저 시작하세요:")
        print(f"  cd serving/api")
        print(f"  uv run uvicorn main:app --reload --port 8000")
        return 1

    # 테스트 실행
    tests = [
        ("피드백 통계 조회", test_feedback_stats),
        ("빈 통계 테스트", test_empty_stats),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAIL {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    # 결과 요약
    print_section("테스트 결과")
    total = passed + failed
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.0f}%)")

    if failed == 0:
        print(f"🎉 모든 테스트 통과!")
        return 0
    else:
        print(f"❌ {failed}개 테스트 실패")
        return 1


if __name__ == "__main__":
    sys.exit(main())
