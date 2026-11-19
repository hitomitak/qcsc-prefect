#!/usr/bin/env bash
# Port matrix: 2 systems x {D0, D1, T0} x 2 repetitions of the ported flow, each in a fresh run
# directory (fresh HOME, PREFECT_HOME, BS-UHF cache and storage), then compare_port.py for each.
# Usage: bash run_port_matrix.sh OUT_ROOT VENV BASELINE_DIR DIAG_UHF
#   BASELINE_DIR  this directory's parent (harness/, inputs/, golden/)
# Exit status 0 only if every run succeeded and every comparison passed.
set -uo pipefail
OUT=$1; VENV=$2; PR0=$3; UHF=$4; H=$(cd "$(dirname "$0")" && pwd)
failures=0
for sys in nh2_sto3g h6_chain2A_sto3g; do
  for cfg in D0 D1 T0; do
    for rep in 1 2; do
      R=$OUT/runs/$sys/$cfg/rep$rep; rm -rf "$R"; mkdir -p "$R"
      env -i PATH=/usr/bin:/bin HOME="$R/home" TMPDIR="${TMPDIR:-/tmp}" PREFECT_HOME="$R/prefect_home" \
        SBD_LOCAL_STORAGE_DIR="$R/storage" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$VENV/bin/python" -I "$H/run_port.py" --harness "$PR0/harness" --system $sys \
        --fcidump "$PR0/inputs/$sys.fcidump" --config $cfg --diag-uhf "$UHF" --run-dir "$R" \
        > "$R.log" 2>&1
      run_rc=$?
      if [ $run_rc -eq 0 ]; then
        cmp=$("$VENV/bin/python" -I "$H/compare_port.py" "$PR0/harness" "$PR0/golden" "$R" $sys $cfg 2>&1)
        cmp_rc=$?
      else
        cmp="not compared: the run failed (see $R.log)"
        cmp_rc=1
      fi
      [ $run_rc -eq 0 ] && [ $cmp_rc -eq 0 ] || failures=$((failures + 1))
      echo "$sys $cfg rep$rep run=$run_rc compare=$cmp_rc $(grep -m1 FLOW_ENERGY "$R.log") | $(echo "$cmp" | head -1)"
      echo "$cmp" | tail -n +2 | head -20
    done
  done
done
echo "port matrix: $failures of 12 runs failed or differ"
[ $failures -eq 0 ]
