#!/usr/bin/env bash
# Self-test of the verification tools: failures must give a non-zero exit status.
# Usage: bash selftest.sh [PORT_VENV PORT_RUN_DIR]
#   Without arguments: checks (1) and (2), which need no solver binary.
#   With a venv that has uhf_sqd installed and one finished port run directory
#   (OUT_ROOT/runs/<system>/<config>/rep<N> from port/run_port_matrix.sh): also check (3).
# (1) port/run_port_matrix.sh exits non-zero when every run fails.
# (2) harness/run_matrix.sh exits non-zero when every run fails.
# (3) port/compare_port.py exits 0 on the run as it is, and 1 after one telemetry energy and one
#     stored array element are changed.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/uhf_sqd_selftest.XXXXXX")
trap 'rm -rf "$T"' EXIT
failures=0
expect() {  # expect <description> <expected: zero|nonzero> <status>
  local ok=1
  if [ "$2" = zero ]; then [ "$3" -eq 0 ] || ok=0; else [ "$3" -ne 0 ] || ok=0; fi
  if [ $ok -eq 1 ]; then echo "PASS  $1 (exit $3)"; else echo "FAIL  $1 (exit $3)"; failures=$((failures + 1)); fi
}

# A Python that always fails stands in for a broken environment.
mkdir -p "$T/stub/bin" "$T/root/venv/bin"
printf '#!/bin/sh\nexit 3\n' > "$T/stub/bin/python"
chmod +x "$T/stub/bin/python"
cp "$T/stub/bin/python" "$T/root/venv/bin/python"

bash "$H/port/run_port_matrix.sh" "$T/port" "$T/stub" "$H" /nonexistent/diag_uhf > "$T/port.log" 2>&1
expect "(1) run_port_matrix.sh with failing runs" nonzero $?
grep -q "12 of 12 runs failed" "$T/port.log" || { echo "FAIL  (1) failures not counted"; failures=$((failures + 1)); }

mkdir -p "$T/root/inputs"
bash "$H/harness/run_matrix.sh" "$T/root" /nonexistent/diag_uhf /nonexistent/diag > "$T/base.log" 2>&1
expect "(2) run_matrix.sh with failing runs" nonzero $?
grep -q "12 of 12 runs failed" "$T/base.log" || { echo "FAIL  (2) failures not counted"; failures=$((failures + 1)); }

if [ $# -ge 2 ]; then
  VENV=$1; RUN=$2
  sys=$(basename "$(dirname "$(dirname "$RUN")")"); cfg=$(basename "$(dirname "$RUN")")
  "$VENV/bin/python" -I "$H/port/compare_port.py" "$H/harness" "$H/golden" "$RUN" "$sys" "$cfg" > /dev/null 2>&1
  expect "(3a) compare_port.py on $sys/$cfg as it is" zero $?
  cp -R "$RUN" "$T/run"
  "$VENV/bin/python" -I - "$T/run" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
r = Path(sys.argv[1])
s = json.loads((r / "summary.json").read_text())
art = s["telemetry"][0]
rows = json.loads(art) if isinstance(art, str) else art
rows[0]["energy"] += 1e-14
s["telemetry"][0] = json.dumps(rows) if isinstance(art, str) else rows
(r / "summary.json").write_text(json.dumps(s, indent=1, default=str))
z = dict(np.load(r / "sqd_data_all.npz"))
k = next(k for k in z if k.endswith("::avg_occupancy"))
a = z[k].copy(); a[0] = np.nextafter(a[0], 2.0); z[k] = a
np.savez(r / "sqd_data_all.npz", **z)
PYEOF
  [ $? -eq 0 ] || { echo "FAIL  (3) could not prepare the tampered run"; failures=$((failures + 1)); }
  "$VENV/bin/python" -I "$H/port/compare_port.py" "$H/harness" "$H/golden" "$T/run" "$sys" "$cfg" > "$T/cmp.log" 2>&1
  expect "(3b) compare_port.py on a tampered copy" nonzero $?
fi
echo "selftest: $failures failure(s)"
[ $failures -eq 0 ]
