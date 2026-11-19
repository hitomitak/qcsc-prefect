#!/usr/bin/env bash
# PR0 baseline matrix: 2 systems x {D0, D1, T0} x 2 repetitions, each in a fresh run directory
# (fresh PREFECT_HOME, BS-UHF cache, storage). Usage: bash run_matrix.sh ROOT DIAG_UHF DIAG_RHF
# Exit status 0 only if every run succeeded.
set -uo pipefail
ROOT=$1; UHF=$2; RHF=$3; H=$(cd "$(dirname "$0")" && pwd)
failures=0
for sys in nh2_sto3g h6_chain2A_sto3g; do
  for cfg in D0 D1 T0; do
    for rep in 1 2; do
      R=$ROOT/runs/$sys/$cfg/rep$rep; rm -rf "$R"; mkdir -p "$R"
      env -i PATH=/usr/bin:/bin HOME="$R/home" TMPDIR="${TMPDIR:-/tmp}" PREFECT_HOME="$R/prefect_home" \
        SBD_TASK_RUNNER=concurrent SBD_UHF_CACHE_DIR="$R/uhf_cache" SBD_LOCAL_STORAGE_DIR="$R/storage" \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$ROOT/venv/bin/python" -I "$H/run_baseline.py" --src "$ROOT/src" --system $sys \
        --fcidump "$ROOT/inputs/$sys.fcidump" --config $cfg --diag-uhf "$UHF" --diag-rhf "$RHF" \
        --run-dir "$R" > "$R.log" 2>&1
      rc=$?
      [ $rc -eq 0 ] || failures=$((failures + 1))
      echo "$sys $cfg rep$rep rc=$rc $(grep -m1 FLOW_ENERGY "$R.log")"
    done
  done
done
echo "baseline matrix: $failures of 12 runs failed"
[ $failures -eq 0 ]
