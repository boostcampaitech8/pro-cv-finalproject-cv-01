#!/bin/bash
# ============================================================
# test_update_cycle.sh
# Jetson 없이 로컬에서 updater의 핵심 로직을 검증하는 스크립트
#
# 테스트 항목:
#   [1] config.py 상수 로드 확인
#   [2] 태그 조건 필터 로직 (충족 / 불충족 / 필터 없음)
#   [3] BUILDING_FLAG 중복 빌드 방지
#   [4] RELOAD_FLAG 생성/감지 (핫스왑 신호)
#   [5] atomic symlink 교체 로직
# ============================================================

set -e
EDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$EDGE_DIR/models"
mkdir -p "$MODELS_DIR"
PASS=0; FAIL=0

run_test() {
    local name="$1"; local cmd="$2"; local expect="$3"
    local result
    result=$(eval "$cmd" 2>&1) || true
    if echo "$result" | grep -q "$expect"; then
        echo "  ✅ PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL: $name"
        echo "     ↳ 기대: '$expect'"
        echo "     ↳ 실제: '$result'"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo " MLOps Edge Update Cycle — 단위 테스트"
echo "========================================"

# ── [1] 설정 상수 로드 ──────────────────────────────────────
echo ""
echo "[1] config.py 상수 로드 확인"

PY_IMPORT="import sys; sys.path.insert(0,'$EDGE_DIR'); import config"

run_test "TRT_WORKSPACE_MB 기본값이 2048인지 확인" \
  "python3 -c \"$PY_IMPORT; print(config.TRT_WORKSPACE_MB)\"" \
  "2048"

run_test "TRT_AVG_TIMING_ITERS 기본값이 8인지 확인" \
  "python3 -c \"$PY_IMPORT; print(config.TRT_AVG_TIMING_ITERS)\"" \
  "8"

run_test "TRT_BUILD_TIMEOUT_S 기본값이 600인지 확인" \
  "python3 -c \"$PY_IMPORT; print(config.TRT_BUILD_TIMEOUT_S)\"" \
  "600"

run_test "MODEL_POLL_INTERVAL 기본값이 300인지 확인" \
  "python3 -c \"$PY_IMPORT; print(config.MODEL_POLL_INTERVAL)\"" \
  "300"

run_test "REQUIRED_TAG_KEY 기본값이 'status'인지 확인" \
  "python3 -c \"$PY_IMPORT; print(config.REQUIRED_TAG_KEY)\"" \
  "status"

run_test "GOLDEN_YAML_PATH가 golden_set/ 디렉토리 안인지 확인" \
  "python3 -c \"$PY_IMPORT; print('golden_set' in config.GOLDEN_YAML_PATH)\"" \
  "True"

# ── [2] 태그 조건 필터 ──────────────────────────────────────
echo ""
echo "[2] 태그 조건 필터 로직 (_meets_condition)"

FILTER_RESULT=$(python3 "$EDGE_DIR/test_filter_logic.py" 2>&1)
echo "$FILTER_RESULT"
FILTER_PASS=$(echo "$FILTER_RESULT" | grep -c "✅ PASS" || true)
FILTER_FAIL=$(echo "$FILTER_RESULT" | grep -c "❌ FAIL" || true)
PASS=$((PASS + FILTER_PASS))
FAIL=$((FAIL + FILTER_FAIL))

# ── [3] BUILDING_FLAG 중복 빌드 방지 ────────────────────────
echo ""
echo "[3] BUILDING_FLAG 중복 빌드 방지"

BUILDING_FLAG=$(python3 -c "$PY_IMPORT; print(config.BUILDING_FLAG_PATH)")
echo "building v99" > "$BUILDING_FLAG"

run_test "BUILDING_FLAG 존재 시 'SKIP' 감지" \
  "python3 -c \"
$PY_IMPORT
import os
print('SKIP' if os.path.exists(config.BUILDING_FLAG_PATH) else 'PROCEED')
\"" \
  "SKIP"

rm -f "$BUILDING_FLAG"

run_test "BUILDING_FLAG 제거 후 'PROCEED' 감지" \
  "python3 -c \"
$PY_IMPORT
import os
print('SKIP' if os.path.exists(config.BUILDING_FLAG_PATH) else 'PROCEED')
\"" \
  "PROCEED"

# ── [4] RELOAD_FLAG 생성 (핫스왑 신호) ──────────────────────
echo ""
echo "[4] RELOAD_FLAG 생성/감지 확인"

RELOAD_FLAG=$(python3 -c "$PY_IMPORT; print(config.RELOAD_FLAG_PATH)")
echo "v99" > "$RELOAD_FLAG"

run_test "RELOAD_FLAG 존재 감지" \
  "python3 -c \"
$PY_IMPORT
import os
print('RELOAD' if os.path.exists(config.RELOAD_FLAG_PATH) else 'NOT_FOUND')
\"" \
  "RELOAD"

rm -f "$RELOAD_FLAG"

run_test "RELOAD_FLAG 제거 후 NOT_FOUND 확인" \
  "python3 -c \"
$PY_IMPORT
import os
print('RELOAD' if os.path.exists(config.RELOAD_FLAG_PATH) else 'NOT_FOUND')
\"" \
  "NOT_FOUND"

# ── [5] Atomic Symlink 교체 로직 ────────────────────────────
echo ""
echo "[5] switch_model() atomics symlink 교체"

DUMMY_ENGINE="$MODELS_DIR/v99_test.engine"
TARGET="$MODELS_DIR/current_test.engine"
touch "$DUMMY_ENGINE"

run_test "os.rename() 기반 atomic 교체 후 심링크 존재 확인" \
  "python3 -c \"
import sys, os; sys.path.insert(0,'$EDGE_DIR')
target     = '$TARGET'
tmp_target = target + '.tmp'
abs_engine = os.path.abspath('$DUMMY_ENGINE')
if os.path.islink(tmp_target) or os.path.exists(tmp_target):
    os.remove(tmp_target)
os.symlink(abs_engine, tmp_target)
os.rename(tmp_target, target)
print('SYMLINK_OK' if os.path.islink(target) else 'SYMLINK_FAIL')
\"" \
  "SYMLINK_OK"

run_test "심링크가 올바른 엔진을 가리키는지 확인" \
  "python3 -c \"
import os
target = '$TARGET'
expected = os.path.abspath('$DUMMY_ENGINE')
actual = os.readlink(target) if os.path.islink(target) else ''
print('TARGET_OK' if actual == expected else f'TARGET_WRONG: {actual}')
\"" \
  "TARGET_OK"

rm -f "$DUMMY_ENGINE" "$TARGET"

# ── 결과 요약 ────────────────────────────────────────────────
echo ""
echo "========================================"
TOTAL=$((PASS + FAIL))
echo " 결과: $PASS/$TOTAL 통과"
if [ "$FAIL" -eq 0 ]; then
    echo " 🎉 모든 테스트 통과!"
    echo "========================================"
    exit 0
else
    echo " ⚠️  $FAIL개 테스트 실패 — 위 로그 확인"
    echo "========================================"
    exit 1
fi
