"""Reduce PR0 run outputs (run_baseline.py layout) to the committed golden set.

Usage: python -I make_golden.py ROOT OUTDIR [--rep 1]
ROOT/runs/<system>/<config>/rep<N>/ must hold summary.json, reference.npz, sqd_data_all.npz,
flow_parameters.json and (for fcidump_records_sha256) solver_root/.

Writes per system and config:
  <system>__<config>.json   flow energy, flow parameters, solver block values, telemetry rows
                            (paths/timestamps dropped), solver job input/output sha256 with
                            davidson_energy strings and fcidump_records_sha256, reference scalars
                            and array hashes, sqd_data content-hash multiset, hashed arrays
  <system>__<config>.npz    per (trial, walker) arrays of the telemetry-referenced sqd_data file:
                            small arrays in full; raw/recovered bitstrings only as sha256 + shape
and once per system:
  <system>__reference.npz   reference arrays (integrals, t2 blocks, initial occupancies)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FULL_KEYS = (
    "ucj_parameter",
    "alphadets",
    "betadets",
    "avg_occupancy",
    "avg_occupancy_b",
    "carryover",
    "carryover_b",
)
HASH_KEYS = ("raw_bitstrings", "recovered_bitstrings")
VOLATILE = {"sqd_data", "last_updated"}


def arr_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def telemetry_rows(summary):
    rows = []
    for art in summary["telemetry"]:
        rows.extend(json.loads(art) if isinstance(art, str) else art)
    return sorted(rows, key=lambda r: (r["trial_index"], r["walker_index"]))


def fcidump_records_sha256(path: Path) -> str:
    """sha256 of the sorted integral records of an FCIDUMP, header excluded (marker-independent)."""
    body, recs = False, []
    for line in path.read_text().splitlines():
        if not body:
            body = "&END" in line
            continue
        if line.strip():
            v, i, j, k, m = line.split()
            recs.append((int(i), int(j), int(k), int(m), repr(float(v))))
    return hashlib.sha256(json.dumps(sorted(recs)).encode()).hexdigest()


def file_groups(npz):
    groups = {}
    for k in npz.files:
        fn, key = k.split("::", 1)
        groups.setdefault(fn, {})[key] = npz[k]
    return groups


def reduce_run(run: Path):
    s = json.loads((run / "summary.json").read_text())
    rows = telemetry_rows(s)
    groups = file_groups(np.load(run / "sqd_data_all.npz"))
    content_hashes = sorted(
        hashlib.sha256(
            "".join(f"{k}:{arr_hash(v)}" for k, v in sorted(g.items())).encode()
        ).hexdigest()
        for g in groups.values()
    )
    arrays, meta_arrays = {}, {}
    for r in rows:
        tag = f"t{r['trial_index']:02d}_w{r['walker_index']}"
        g = groups[Path(r["sqd_data"].removeprefix("file://")).name]
        for k in FULL_KEYS:
            if k in g:
                arrays[f"{tag}::{k}"] = g[k]
        for k in HASH_KEYS:
            if k in g:
                meta_arrays[f"{tag}::{k}"] = {
                    "sha256": arr_hash(g[k]),
                    "shape": list(g[k].shape),
                    "dtype": g[k].dtype.str,
                }
    records = {}
    if (run / "solver_root").is_dir():
        for j in (run / "solver_root").glob("job_*"):
            if (j / "fcidump.txt").exists():
                records[j.name] = fcidump_records_sha256(j / "fcidump.txt")
    jobs = []
    for j in s["solver_jobs"]:
        rec = {k: v for k, v in j.items() if k != "dir"}
        if j["dir"] in records:
            rec["fcidump_records_sha256"] = records[j["dir"]]
        jobs.append(rec)
    ref = np.load(run / "reference.npz")
    golden = {
        "source": "kunalroy332/qcsc-prefect@3968e585f7ca8137c127e5ee9f4fb383322d3d0d",
        "system": s["system"],
        "config": s["config"],
        "flow_energy": s["flow_energy"],
        # The FCIDUMP path is run-specific; keep the file name (content: inputs/inputs.json sha256).
        "flow_parameters": {
            **(fp := json.loads((run / "flow_parameters.json").read_text())),
            "fcidump": Path(fp["fcidump"]).name if fp.get("fcidump") else None,
        },
        "solver_block": s["solver_block"],
        "reference": {
            k: s[k]
            for k in (
                "ecore",
                "norb",
                "nelec",
                "E_HF_from_integrals_plus_ecore",
                "E_UHF_cache",
                "reference_S2",
            )
        }
        | {"array_sha256": {k: arr_hash(ref[k]) for k in sorted(ref.files)}},
        "telemetry": [{k: v for k, v in r.items() if k not in VOLATILE} for r in rows],
        "solver_jobs": sorted(json.dumps(j, sort_keys=True) for j in jobs),
        "sqd_data_content_sha256": content_hashes,
        "hashed_arrays": meta_arrays,
    }
    return golden, arrays, {k: ref[k] for k in ref.files}


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rep = sys.argv[sys.argv.index("--rep") + 1] if "--rep" in sys.argv else "1"
    out.mkdir(parents=True, exist_ok=True)
    for sysdir in sorted((root / "runs").iterdir()):
        if not sysdir.is_dir():
            continue
        for cfg in ("D0", "D1", "T0"):
            golden, arrays, ref = reduce_run(sysdir / cfg / f"rep{rep}")
            (out / f"{sysdir.name}__{cfg}.json").write_text(json.dumps(golden, indent=1))
            np.savez_compressed(out / f"{sysdir.name}__{cfg}.npz", **arrays)
        np.savez_compressed(out / f"{sysdir.name}__reference.npz", **ref)
    print("golden written to", out)


if __name__ == "__main__":
    main()
