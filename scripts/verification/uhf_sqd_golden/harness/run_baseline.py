"""PR0 baseline: run the UNMODIFIED 3968e58 flow `riken_sqd_de` and record its outputs.

Usage (one run per process; fresh run directory):
    python -I run_baseline.py --src SRC --system NAME --fcidump F --config {D0,D1,T0} \
        --diag-uhf BIN --diag-rhf BIN --run-dir DIR

Configs (only in-scope features; everything not listed keeps the 3968e58 default):
  D0  original defaults: FlowParameters(fcidump, quantum_source="random"); solver blocks from
      create_blocks.py defaults (block 4, iteration 1, tolerance 1e-2, carryover_ratio 0.1,
      carryover_type 0) with --method uhf --solver-mode cpu --hpc-target local.
  D1  coverage of the in-scope code paths with the same original solver defaults:
      sqd_dim=400 (subsampling truncates), n_recovery_steps=2, n_batches=2,
      de_params.iterations=2 (DE mutation/crossover/selection), quantum_source="random".
  T0  FCI comparison: D0 flow parameters (default sqd_dim -> whole space for these systems) and
      a solver block with tight Davidson settings (block 30, iteration 60, tolerance 1e-10);
      other solver values as create_blocks defaults.
Environment set by the caller: PREFECT_HOME, SBD_TASK_RUNNER=concurrent, SBD_UHF_CACHE_DIR,
SBD_LOCAL_STORAGE_DIR, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1.
Nothing in SRC is modified; create_blocks.py is executed from SRC with runpy.
"""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inprocess_prefect  # noqa: E402,F401  (must precede Prefect imports below)
import numpy as np  # noqa: E402

TIGHT = ["--block", "30", "--iteration", "60", "--tolerance", "1e-10"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in (
        "--src",
        "--system",
        "--fcidump",
        "--config",
        "--diag-uhf",
        "--diag-rhf",
        "--run-dir",
    ):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    solver_root = run / "solver_root"

    # 1. Blocks, with the branch's own CLI (unmodified), local target, UHF CPU binary.
    argv = [
        "create_blocks.py",
        "--hpc-target",
        "local",
        "--work-dir",
        str(solver_root),
        "--sbd-executable",
        a.diag_rhf,
        "--sbd-executable-uhf",
        a.diag_uhf,
        "--method",
        "uhf",
        "--solver-mode",
        "cpu",
    ]
    if a.config == "T0":
        argv += TIGHT
    sys.argv = argv
    runpy.run_path(str(Path(a.src) / "algorithms/sbd/create_blocks.py"), run_name="__main__")

    # 2. Flow, unmodified.
    from sbd.flow_params import FlowParameters
    from sbd.main import riken_sqd_de

    params = {"fcidump": str(Path(a.fcidump).resolve()), "quantum_source": "random"}
    if a.config == "D1":
        params.update(sqd_dim=400, n_recovery_steps=2, n_batches=2, de_params={"iterations": 2})
    fp = FlowParameters(**params)
    (run / "flow_parameters.json").write_text(fp.model_dump_json(indent=1))
    energy = riken_sqd_de(parameters=fp)

    # 3. Outputs.
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey

    with get_client(sync_client=True) as client:
        arts = client.read_artifacts(
            artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=["sqd-telemetry"]))
        )
        blocks = {b.name: b.data for b in client.read_block_documents()}
    telemetry = [art.data for art in arts]

    storage = Path(__import__("os").environ["SBD_LOCAL_STORAGE_DIR"])
    sqd_files = sorted(storage.rglob("*.npz"))
    arrays = {}
    for f in sqd_files:
        with np.load(f, allow_pickle=False) as z:
            arrays[f.name] = {k: z[k] for k in z.files}
    np.savez(
        run / "sqd_data_all.npz",
        **{f"{fn}::{k}": v for fn, d in arrays.items() for k, v in d.items()},
    )

    jobs = []
    for j in sorted(solver_root.rglob("job_*")):
        rec = {"dir": j.name}
        for n in (
            "fcidump.txt",
            "AlphaDets.bin",
            "BetaDets.bin",
            "carryover.bin",
            "carryover_b.bin",
            "occ_a.txt",
            "occ_b.txt",
        ):
            if (j / n).exists():
                rec[n] = sha(j / n)
        if (j / "davidson_energy.txt").exists():
            rec["davidson_energy"] = (j / "davidson_energy.txt").read_text().strip()
        jobs.append(rec)

    # 4. Reference stage, same branch function (cache written by the flow is read back).
    import logging

    from qcsc_workflow_utility import chem

    chem.get_run_logger = lambda: logging.getLogger("chem")  # outside a flow context only
    fn = getattr(
        chem.compute_molecular_integrals_from_fcidump,
        "fn",
        chem.compute_molecular_integrals_from_fcidump,
    )
    ep = fn(fcidump_file=str(Path(a.fcidump).resolve()), unrestricted=True)
    ref = {
        k: np.asarray(getattr(ep, k))
        for k in (
            "one_body_tensor",
            "one_body_tensor_b",
            "two_body_tensor",
            "two_body_tensor_ab",
            "two_body_tensor_bb",
            "t2",
            "t2_ab",
            "t2_bb",
        )
    }
    ref["initial_occupancy_a"], ref["initial_occupancy_b"] = (
        np.asarray(x) for x in ep.initial_occupancy
    )
    np.savez(run / "reference.npz", **ref)
    na, nb = ep.num_electrons
    h1a, h1b, gaa, gab, gbb = (
        ref[k]
        for k in (
            "one_body_tensor",
            "one_body_tensor_b",
            "two_body_tensor",
            "two_body_tensor_ab",
            "two_body_tensor_bb",
        )
    )
    e_hf = (
        np.trace(h1a[:na, :na])
        + np.trace(h1b[:nb, :nb])
        + 0.5
        * (np.einsum("iijj", gaa[:na, :na, :na, :na]) - np.einsum("ijji", gaa[:na, :na, :na, :na]))
        + 0.5
        * (np.einsum("iijj", gbb[:nb, :nb, :nb, :nb]) - np.einsum("ijji", gbb[:nb, :nb, :nb, :nb]))
        + np.einsum("iijj", gab[:na, :na, :nb, :nb])
    )
    cache = sorted(Path(__import__("os").environ["SBD_UHF_CACHE_DIR"]).glob("*.npz"))
    s2 = None
    if cache:
        from pyscf.scf.uhf import spin_square

        with np.load(cache[0]) as c:
            mo_a, mo_b = c["mo_a"], c["mo_b"]
            s2 = float(spin_square((mo_a[:, :na], mo_b[:, :nb]), np.eye(mo_a.shape[0]))[0])
            e_tot_cache = float(c["e_tot"])
    summary = {
        "system": a.system,
        "config": a.config,
        "flow_energy": energy,
        "ecore": float(ep.nuclear_repulsion_energy),
        "norb": ep.num_orbitals,
        "nelec": [na, nb],
        "E_HF_from_integrals_plus_ecore": float(e_hf + ep.nuclear_repulsion_energy),
        "E_UHF_cache": e_tot_cache if cache else None,
        "reference_S2": s2,
        "reference_hash": hashlib.sha256(
            b"".join(np.ascontiguousarray(ref[k]).tobytes() for k in sorted(ref))
        ).hexdigest(),
        "sqd_data_files": [f.name for f in sqd_files],
        "solver_jobs": jobs,
        "telemetry": telemetry,
        "solver_block": {
            k: v
            for k, v in blocks.get("davidson-solver", {}).items()
            if k
            in (
                "block",
                "iteration",
                "tolerance",
                "carryover_ratio",
                "carryover_type",
                "do_rdm",
                "method",
                "solver_mode",
            )
        },
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("FLOW_ENERGY", repr(energy))


if __name__ == "__main__":
    main()
