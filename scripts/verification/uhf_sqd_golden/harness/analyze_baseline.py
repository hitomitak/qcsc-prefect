"""Analyse the PR0 baseline matrix: repetition equality and the T0 full-space FCI comparison.

Usage: python -I analyze_baseline.py ROOT
Uses only numpy/pyscf (no branch code). FCI reference: pyscf.fci.direct_spin1 on the restricted
FCIDUMP integrals (full space; independent of the UHF transform and of SBD), plus ECORE.
"""

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from pyscf import fci
from pyscf.tools import fcidump

root = Path(sys.argv[1])
VOLATILE = {"sqd_data", "last_updated"}  # run-specific paths and timestamps


def telemetry_rows(summary):
    rows = []
    for art in summary["telemetry"]:
        data = json.loads(art) if isinstance(art, str) else art
        rows.extend(data)
    return sorted(rows, key=lambda r: (r["trial_index"], r["walker_index"]))


def strip(row):
    return {k: v for k, v in row.items() if k not in VOLATILE}


def walker_arrays(run: Path, summary):
    """sqd_data arrays keyed by (trial, walker), matched through the telemetry sqd_data path."""
    allz = np.load(run / "sqd_data_all.npz")
    out = {}
    for r in telemetry_rows(summary):
        fname = Path(r["sqd_data"].removeprefix("file://")).name
        out[(r["trial_index"], r["walker_index"])] = {
            k.split("::", 1)[1]: allz[k] for k in allz.files if k.startswith(fname + "::")
        }
    return out


def same_arrays(a, b):
    if a.keys() != b.keys():
        return False
    for key in a:
        if a[key].keys() != b[key].keys():
            return False
        for k in a[key]:
            x, y = a[key][k], b[key][k]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                return False
    return True


inputs = json.loads((root / "inputs" / "inputs.json").read_text())
report = {}
for system in ("nh2_sto3g", "h6_chain2A_sto3g"):
    d = fcidump.read(str(root / "inputs" / f"{system}.fcidump"), verbose=False)
    norb, ms2, nel = d["NORB"], d.get("MS2", 0), d["NELEC"]
    nelec = ((nel + ms2) // 2, (nel - ms2) // 2)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    e_fci = solver.kernel(d["H1"], d["H2"], norb, nelec, ecore=d["ECORE"])[0]
    full_dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    for cfg in ("D0", "D1", "T0"):
        runs = [root / "runs" / system / cfg / f"rep{i}" for i in (1, 2)]
        s = [json.loads((r / "summary.json").read_text()) for r in runs]
        t = [[strip(x) for x in telemetry_rows(si)] for si in s]
        jobs = [
            sorted(
                json.dumps({k: v for k, v in j.items() if k != "dir"}, sort_keys=True)
                for j in si["solver_jobs"]
            )
            for si in s
        ]
        ref = [np.load(r / "reference.npz") for r in runs]
        same_ref = all(ref[0][k].tobytes() == ref[1][k].tobytes() for k in ref[0].files)
        rec = {
            "flow_energy": s[0]["flow_energy"],
            "rep_equal": {
                "flow_energy": s[0]["flow_energy"] == s[1]["flow_energy"],
                "reference_arrays": same_ref,
                "reference_hash": s[0]["reference_hash"] == s[1]["reference_hash"],
                "telemetry_rows": t[0] == t[1],
                "walker_arrays": same_arrays(
                    walker_arrays(runs[0], s[0]), walker_arrays(runs[1], s[1])
                ),
                "solver_job_inputs_outputs": jobs[0] == jobs[1],
            },
            "n_walker_records": len(t[0]),
            "n_solver_jobs": len(s[0]["solver_jobs"]),
            "net_subspace_dims": sorted({r["net_subspace_dim"] for r in t[0]}),
            "full_space_dim": full_dim,
            "reference": {
                k: s[0][k]
                for k in (
                    "ecore",
                    "E_HF_from_integrals_plus_ecore",
                    "E_UHF_cache",
                    "reference_S2",
                    "reference_hash",
                )
            },
            "solver_block": s[0]["solver_block"],
        }
        if cfg == "T0":
            all_full = all(r["net_subspace_dim"] == full_dim for r in t[0])
            rec["fci"] = {
                "E_FCI_pyscf_restricted": e_fci,
                "all_walkers_full_space": all_full,
                "flow_minus_fci": s[0]["flow_energy"] - e_fci,
                "walker_energy_minus_fci": [r["energy"] - e_fci for r in t[0]],
                "tolerance": 1e-8,
                "pass": all_full and abs(s[0]["flow_energy"] - e_fci) < 1e-8,
            }
        else:
            rec["flow_minus_fci_info_only"] = s[0]["flow_energy"] - e_fci
        report[f"{system}/{cfg}"] = rec
(root / "baseline_report.json").write_text(json.dumps(report, indent=1, default=float))
for k, v in report.items():
    print(
        k,
        "E=%.12f" % v["flow_energy"],
        "rep_equal=%s" % all(v["rep_equal"].values()),
        {kk: vv for kk, vv in v["rep_equal"].items() if not vv} or "",
        "dims",
        v["net_subspace_dims"],
        "/",
        v["full_space_dim"],
        ("FCI diff %.2e pass=%s" % (v["fci"]["flow_minus_fci"], v["fci"]["pass"]))
        if "fci" in v
        else "(vs FCI %+.2e, info)" % v["flow_minus_fci_info_only"],
        "S2=%.4f" % v["reference"]["reference_S2"],
    )
