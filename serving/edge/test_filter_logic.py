#!/usr/bin/env python3
"""
test_filter_logic.py
test_update_cycle.sh의 [2] 섹션에서 호출하는 조건 필터 단위 테스트.
각 케이스별로 PASS/FAIL을 출력하고, 실패 시 exit code 1로 종료합니다.
"""
import sys

PASS = 0
FAIL = 0


def meets_condition(meta: dict, key: str, value: str) -> bool:
    """updater.py의 _meets_condition()과 동일한 로직"""
    if not key:
        return True
    tags = meta.get("tags", {})
    return tags.get(key) == value


def run_test(name: str, actual: bool, expected: bool):
    global PASS, FAIL
    if actual == expected:
        print(f"  ✅ PASS: {name}")
        PASS += 1
    else:
        print(f"  ❌ FAIL: {name}")
        print(f"     ↳ 기대: {expected},  실제: {actual}")
        FAIL += 1


# 케이스 1: 조건 충족
run_test(
    "조건 충족: status=retrained → True",
    meets_condition({"tags": {"status": "retrained"}}, "status", "retrained"),
    True
)

# 케이스 2: 조건 불충족 (값 다름)
run_test(
    "조건 불충족: status=staging → False",
    meets_condition({"tags": {"status": "staging"}}, "status", "retrained"),
    False
)

# 케이스 3: 태그 키 자체 누락
run_test(
    "태그 키 누락 → False",
    meets_condition({"tags": {}}, "status", "retrained"),
    False
)

# 케이스 4: REQUIRED_TAG_KEY가 빈 문자열 → 항상 허용
run_test(
    "REQUIRED_TAG_KEY 빈 문자열 → 항상 True",
    meets_condition({"tags": {}}, "", "retrained"),
    True
)

total = PASS + FAIL
print(f"\n  필터 테스트 결과: {PASS}/{total} 통과")
sys.exit(0 if FAIL == 0 else 1)
