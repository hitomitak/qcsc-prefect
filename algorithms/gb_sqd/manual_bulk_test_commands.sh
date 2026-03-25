#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/hitomi/Project/qcsc-prefect/algorithms/gb_sqd"
PYTHON_BIN="${PYTHON_BIN:-python}"

SEED_DIR="/vol0002/mdt6/data/ra010014/u13450/gb_data/test_dat/13_18MO_Wat/atom_10012"
WORKSPACE_ROOT="/vol0002/mdt6/data/ra010014/u13450/sbd_jobs"

run() {
  echo "+ $*"
  "$@"
}

cd "${PROJECT_DIR}"

echo "== Prepare workspace =="
run "${PYTHON_BIN}" manual_bulk_test.py prepare \
  --seed-dir "${SEED_DIR}" \
  --workspace-root "${WORKSPACE_ROOT}"

echo
echo "== List scenarios =="
run "${PYTHON_BIN}" manual_bulk_test.py list

echo
echo "== Describe scenarios =="
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario1_ext_happy
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario2_trim_happy
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario3_retry_runtime
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario4_override_rerun
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario5_skip_completed_rerun
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario6_mixed_failure
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario7_user_queue
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario8_flow_jobs_only
run "${PYTHON_BIN}" manual_bulk_test.py describe \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario9_invalid_override

echo
echo "== Scenario 1: ExtSQD happy path =="
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario1_ext_happy

echo
echo "== Scenario 2: TrimSQD happy path =="
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario2_trim_happy

echo
echo "== Scenario 3: Retry on runtime failure =="
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario3_retry_runtime

echo
echo "== Scenario 4: Repair failed target then rerun with target_overrides =="
run "${PYTHON_BIN}" manual_bulk_test.py repair \
  --seed-dir "${SEED_DIR}" \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario3_retry_runtime \
  --relative-target-path retry_bad_fcidump/atom_0003
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario4_override_rerun

echo
echo "== Scenario 5: skip_completed rerun =="
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario5_skip_completed_rerun

echo
echo "== Scenario 6: Mixed success/failure =="
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario6_mixed_failure

echo
echo "== Scenario 7: user_queue throttling =="
echo "Prepare unrelated jobs in the same Fugaku queue before running this scenario."
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario7_user_queue

echo
echo "== Scenario 8: flow_jobs_only throttling =="
echo "Keep unrelated jobs in the same queue; this scenario should ignore them."
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario8_flow_jobs_only

echo
echo "== Scenario 9: Invalid override validation =="
run "${PYTHON_BIN}" manual_bulk_test.py run \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scenario scenario9_invalid_override

echo
echo "Manual bulk test sequence finished."
