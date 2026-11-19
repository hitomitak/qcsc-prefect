"""Compare a run (run_baseline.py output layout) against the PR0 golden set.

Usage: python -I compare_golden.py GOLDEN_DIR RUN_DIR SYSTEM CONFIG [--fcidump-records]
Exit 0 if identical, 1 otherwise; prints the differences (first 50).

Rules (see PR0_BASELINE.md, "Comparison method"):
- exact equality everywhere: float64 values compared with ==, arrays by dtype/shape/bytes, hashes
  by string; JSON floats are Python repr (round-trip exact);
- run-specific fields (sqd_data path, last_updated, solver job directory names) are ignored;
- sqd_data files are matched by content hash (their file names are random);
- --fcidump-records: compare the solver FCIDUMP inputs by their parsed integral records
  (fcidump_records_sha256) instead of the whole-file sha256. Use it for the port, whose FCIDUMP
  header carries an extra layout-marker token. Needs the run's solver_root directory.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_golden import reduce_run  # noqa: E402


def diff(path, a, b, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing on {'golden' if k not in a else 'run'} side")
            else:
                diff(f"{path}.{k}", a[k], b[k], out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(f"{path}[{i}]", x, y, out)
    elif a != b:
        out.append(f"{path}: golden={a!r} run={b!r}")


def main():
    gdir, run, system, cfg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    golden = json.loads((gdir / f"{system}__{cfg}.json").read_text())
    garr = np.load(gdir / f"{system}__{cfg}.npz")
    gref = np.load(gdir / f"{system}__reference.npz")
    new, arrays, ref = reduce_run(run)
    out = []
    if "--fcidump-records" in sys.argv:

        def drop(jobs):
            return sorted(
                json.dumps(
                    {k: v for k, v in json.loads(j).items() if k != "fcidump.txt"}, sort_keys=True
                )
                for j in jobs
            )

        golden["solver_jobs"] = drop(golden["solver_jobs"])
        new["solver_jobs"] = drop(new["solver_jobs"])
    diff("$", golden, new, out)
    for k in sorted(set(garr.files) | set(arrays)):
        if k not in garr.files or k not in arrays:
            out.append(f"array {k}: missing on {'golden' if k not in garr.files else 'run'} side")
            continue
        x, y = garr[k], arrays[k]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            out.append(f"array {k}: differs")
    for k in gref.files:
        if k not in ref or gref[k].tobytes() != np.ascontiguousarray(ref[k]).tobytes():
            out.append(f"reference array {k}: differs")
    print(f"{system}/{cfg}: {'IDENTICAL' if not out else f'{len(out)} DIFFERENCES'}")
    for line in out[:50]:
        print("  ", line)
    sys.exit(0 if not out else 1)


if __name__ == "__main__":
    main()
