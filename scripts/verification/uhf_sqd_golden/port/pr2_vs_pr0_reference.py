"""PR2: the ported UHF reference against the PR0 record of the source (3968e58).

Usage: python -I pr2_vs_pr0_reference.py BASELINE_DIR PR0_RUNS WORK_DIR
  BASELINE_DIR  this directory's parent (golden/, inputs/)
  PR0_RUNS      ROOT/runs of a baseline run with harness/run_matrix.sh (it keeps uhf_cache/)
Run in an environment with the PR0 third-party pins, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1.

For each system, three paths of compute_uhf_integrals_from_fcidump:
  A  fresh: empty cache directory -> broken-symmetry SCF; the written cache file must have
     the PR0 file name and bitwise-equal contents;
  B  reuse: the cache written by A (how PR0 captured its reference);
  C  source cache: the cache file the source wrote in PR0, copied to a new directory.
Each path's arrays are compared bitwise (dtype, shape, bytes) with golden/<system>__reference.npz,
and its scalars with golden/<system>__D0.json "reference". Exit 0 only if everything is equal.
"""

import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from qcsc_workflow_utility import uhf

logging.basicConfig(level=logging.WARNING)
pr0, runs, work = (Path(a).resolve() for a in sys.argv[1:4])
work.mkdir(parents=True, exist_ok=True)
SYSTEMS = {
    "nh2_sto3g": "cba014e4b5ea9c1a218651988f60f1bb43406d1f147140269a4caaa6130cb87f",
    "h6_chain2A_sto3g": "352de236a0f64e171df67e9b3415ed4f231f0cfc50c1514bf3a17ebbb51a29e2",
}
ARRAYS = {
    "one_body_tensor": "one_body_tensor",
    "one_body_tensor_b": "one_body_tensor_b",
    "two_body_tensor": "two_body_tensor",
    "two_body_tensor_ab": "two_body_tensor_ab",
    "two_body_tensor_bb": "two_body_tensor_bb",
    "t2": "t2",
    "t2_ab": "t2_ab",
    "t2_bb": "t2_bb",
}
CACHE_KEYS = ("mo_a", "mo_b", "occ_a", "occ_b", "ene_a", "ene_b", "e_tot", "nelec")
failures = []
report = {}


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def e_hf(ep):
    # The PR0 harness formula (run_baseline.py), on the port's arrays.
    na, nb = ep.num_electrons
    h1a, h1b = ep.one_body_tensor, ep.one_body_tensor_b
    gaa, gab, gbb = ep.two_body_tensor, ep.two_body_tensor_ab, ep.two_body_tensor_bb
    e = (
        np.trace(h1a[:na, :na])
        + np.trace(h1b[:nb, :nb])
        + 0.5
        * (np.einsum("iijj", gaa[:na, :na, :na, :na]) - np.einsum("ijji", gaa[:na, :na, :na, :na]))
        + 0.5
        * (np.einsum("iijj", gbb[:nb, :nb, :nb, :nb]) - np.einsum("ijji", gbb[:nb, :nb, :nb, :nb]))
        + np.einsum("iijj", gab[:na, :na, :nb, :nb])
    )
    return float(e + ep.nuclear_repulsion_energy)


def check(system, path, ep, gref, gjson):
    tag = f"{system}/{path}"
    rows = {}
    for field, key in ARRAYS.items():
        rows[key] = same(getattr(ep, field), gref[key])
    rows["initial_occupancy_a"] = same(ep.initial_occupancy[0], gref["initial_occupancy_a"])
    rows["initial_occupancy_b"] = same(ep.initial_occupancy[1], gref["initial_occupancy_b"])
    r = gjson["reference"]
    rows["ecore"] = ep.nuclear_repulsion_energy == r["ecore"]
    rows["norb"] = ep.num_orbitals == r["norb"]
    rows["nelec"] = list(ep.num_electrons) == r["nelec"]
    rows["E_UHF (reference_energy == E_UHF_cache)"] = ep.reference_energy == r["E_UHF_cache"]
    rows["E_HF from integrals + ECORE"] = e_hf(ep) == r["E_HF_from_integrals_plus_ecore"]
    s2_diff = ep.reference_spin_square - r["reference_S2"]
    for k, ok in rows.items():
        if not ok:
            failures.append(f"{tag}: {k}")
    report[tag] = {
        "all_equal": all(rows.values()),
        "checks": rows,
        "reference_spin_square": ep.reference_spin_square,
        "golden_reference_S2": r["reference_S2"],
        "S2_difference": s2_diff,
        "reference_cache_key": ep.reference_cache_key,
    }
    print(
        f"{tag}: {'IDENTICAL' if all(rows.values()) else 'DIFFERS'} "
        f"({sum(rows.values())}/{len(rows)}), <S^2> port-golden = {s2_diff:+.3e}"
    )


for system, sha in SYSTEMS.items():
    fcidump = pr0 / "inputs" / f"{system}.fcidump"
    assert hashlib.sha256(fcidump.read_bytes()).hexdigest() == sha, fcidump
    gref = np.load(pr0 / "golden" / f"{system}__reference.npz")
    gjson = json.loads((pr0 / "golden" / f"{system}__D0.json").read_text())
    (src_cache,) = sorted((runs / system / "D0" / "rep1" / "uhf_cache").glob("uhf_bs_*.npz"))

    # A: fresh
    cache_a = work / system / "cache_a"
    shutil.rmtree(cache_a, ignore_errors=True)
    t0 = time.perf_counter()
    ep = uhf.compute_uhf_integrals_from_fcidump.fn(str(fcidump), cache_dir=str(cache_a))
    t_a = time.perf_counter() - t0
    (written,) = sorted(cache_a.glob("uhf_bs_*.npz"))
    if written.name != src_cache.name:
        failures.append(f"{system}/A: cache file name {written.name} != PR0 {src_cache.name}")
    with np.load(written) as a, np.load(src_cache) as b:
        for k in CACHE_KEYS:
            if not same(a[k], b[k]):
                failures.append(f"{system}/A: cache {k} differs from PR0")
        if sorted(a.files) != sorted(b.files):
            failures.append(f"{system}/A: cache keys {sorted(a.files)} != PR0 {sorted(b.files)}")
    print(f"{system}/A: cache file {written.name} (PR0 {src_cache.name}), SCF+UCCSD {t_a:.1f} s")
    check(system, "A fresh", ep, gref, gjson)

    # B: reuse the cache written by A; a second SCF would be a failure of the cache path.
    real = uhf._converge_broken_symmetry_uhf
    uhf._converge_broken_symmetry_uhf = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("SCF ran")
    )
    try:
        ep_b = uhf.compute_uhf_integrals_from_fcidump.fn(str(fcidump), cache_dir=str(cache_a))
        check(system, "B reuse", ep_b, gref, gjson)
        # C: the cache file the source wrote in PR0
        cache_c = work / system / "cache_c"
        shutil.rmtree(cache_c, ignore_errors=True)
        cache_c.mkdir(parents=True)
        shutil.copy2(src_cache, cache_c / src_cache.name)
        ep_c = uhf.compute_uhf_integrals_from_fcidump.fn(str(fcidump), cache_dir=str(cache_c))
        check(system, "C source cache", ep_c, gref, gjson)
    finally:
        uhf._converge_broken_symmetry_uhf = real

(work / "pr2_vs_pr0_reference.json").write_text(
    json.dumps({"failures": failures, "report": report}, indent=1, default=str)
)
if failures:
    print("DIFFERENCES:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("ALL IDENTICAL to the PR0 record (arrays bitwise, scalars exact, cache name and contents)")
