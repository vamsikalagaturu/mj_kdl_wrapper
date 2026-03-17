#!/usr/bin/env bash
# run_tests.sh — build and run all mj_kdl_wrapper tests.
#
# Usage:
#   ./run_tests.sh [build_dir]
#
# Default build dir: build/
# Pass a custom dir to test a release build, e.g.:
#   ./run_tests.sh build-release

set -euo pipefail

BUILD_DIR="${1:-build}"

# ANSI colours
RED='\033[31m'
GRN='\033[32m'
YLW='\033[33m'
GRY='\033[90m'
RST='\033[0m'

PASS=0
FAIL=0
SKIP=0
TOTAL=0

run_test() {
    local name="$1"
    local bin="$BUILD_DIR/$name"
    ((TOTAL++)) || true

    if [[ ! -x "$bin" ]]; then
        echo -e "${GRY}[ SKIP ]${RST} $name  (binary not found in $BUILD_DIR/)"
        ((SKIP++)) || true
        return
    fi

    local out rc=0
    out=$("$bin" 2>&1) || rc=$?

    if [[ $rc -ne 0 ]]; then
        echo -e "${RED}[ FAIL ]${RST} $name  (exit $rc)"
        echo "$out" | sed 's/^/         /'
        ((FAIL++)) || true
    elif echo "$out" | grep -q '^\[.*SKIP.*\]'; then
        echo -e "${GRY}[ SKIP ]${RST} $name"
        ((SKIP++)) || true
    else
        echo -e "${GRN}[ PASS ]${RST} $name"
        ((PASS++)) || true
    fi
}

echo "Running mj_kdl_wrapper tests from '$BUILD_DIR/'..."
echo ""

run_test test_init
run_test test_velocity
run_test test_gravity_comp
run_test test_dual_arm
run_test test_table_scene
run_test test_kinova_gen3_menagerie
run_test test_kinova_gen3_gripper
run_test test_kinova_gen3_impedance
run_test test_kinova_gen3_pick

echo ""
echo "────────────────────────────────"
echo -e "Results  PASS: ${GRN}$PASS${RST}  FAIL: ${RED}$FAIL${RST}  SKIP: ${GRY}$SKIP${RST}  TOTAL: $TOTAL"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
