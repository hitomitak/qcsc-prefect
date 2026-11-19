#!/usr/bin/env bash
# Build the golden set from rep1 of a PR0 run root, compare rep2 against it (whole-file and
# --fcidump-records modes) and run a tamper test that must be detected.
# Exit status 0 only if every comparison is identical and the tamper test is detected.
# Usage: bash golden_check.sh ROOT HARNESS_DIR GOLDEN_OUT
#   ROOT        run root used with run_matrix.sh (ROOT/venv, ROOT/runs)
#   HARNESS_DIR directory holding make_golden.py and compare_golden.py
#   GOLDEN_OUT  output directory for the golden set (recreated)
set -uo pipefail
ROOT=$1; H=$2; G=$3
PY=("$ROOT/venv/bin/python" -I)
failures=0
rm -rf "$G" && "${PY[@]}" "$H/make_golden.py" "$ROOT" "$G" --rep 1 && du -sh "$G" || exit 1
for s in nh2_sto3g h6_chain2A_sto3g; do for c in D0 D1 T0; do
  for mode in "" --fcidump-records; do
    out=$("${PY[@]}" "$H/compare_golden.py" "$G" "$ROOT/runs/$s/$c/rep2" $s $c $mode 2>&1)
    rc=$?
    [ $rc -eq 0 ] || failures=$((failures + 1))
    echo "${mode:+  [records mode] }$(echo "$out" | head -1)"
  done
done; done
T=$(mktemp -d "${TMPDIR:-/tmp}/golden_tamper.XXXXXX")
cp -R "$ROOT/runs/nh2_sto3g/D1/rep2" "$T/run" || { echo "tamper setup failed"; exit 1; }
"${PY[@]}" - "$T/run" <<'PYEOF' || { echo "tamper setup failed"; exit 1; }
import json, sys
from pathlib import Path
import numpy as np
r = Path(sys.argv[1])
s = json.loads((r / "summary.json").read_text()); s["flow_energy"] += 1e-14
(r / "summary.json").write_text(json.dumps(s, indent=1, default=str))
z = dict(np.load(r / "sqd_data_all.npz")); k = next(k for k in z if k.endswith("::alphadets"))
a = z[k].copy(); a[[1, 2]] = a[[2, 1]]; z[k] = a; np.savez(r / "sqd_data_all.npz", **z)
f = sorted((r / "solver_root").glob("job_*"))[0] / "fcidump.txt"
t = f.read_text().splitlines()
v, *idx = t[5].split(); t[5] = " ".join([repr(float(v) + 1e-12), *idx]); f.write_text("\n".join(t) + "\n")
PYEOF
echo "--- tamper test (energy +1e-14, two alphadets swapped, one FCIDUMP record +1e-12); must FAIL:"
out=$("${PY[@]}" "$H/compare_golden.py" "$G" "$T/run" nh2_sto3g D1 --fcidump-records 2>&1)
rc=$?
echo "$out" | head -4
echo "tamper exit=$rc (expected 1)"
[ $rc -eq 1 ] || failures=$((failures + 1))
rm -rf "$T"
echo "golden check: $failures failure(s)"
[ $failures -eq 0 ]
