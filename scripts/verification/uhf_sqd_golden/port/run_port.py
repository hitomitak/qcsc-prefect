"""PR3: run the ported flow uhf_sqd_de with the PR0 inputs, settings and seeds.

Usage (one run per process; fresh run directory):
    python -I run_port.py --harness PR0_HARNESS --system NAME --fcidump F --config {D0,D1,T0} \
        --diag-uhf BIN --run-dir DIR
Writes the PR0 run layout (run_baseline.py): flow_parameters.json, summary.json, reference.npz,
sqd_data_all.npz and solver_root/. Environment set by the caller: PREFECT_HOME,
SBD_LOCAL_STORAGE_DIR, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1.

Settings (PR0_BASELINE.md "Configurations"):
  D0  FlowParameters(fcidump, quantum_source=<random QuantumSourceBlock>), the source's solver
      defaults (block 4, iteration 1, tolerance 1e-2, carryover_ratio 0.1, carryover_type 0).
  D1  D0 plus sqd_dim=400, n_recovery_steps=2, n_batches=2, de_params.iterations=2.
  T0  D0 flow parameters, solver block 30, iteration 60, tolerance 1e-10.
The BS-UHF reference cache is enabled with an empty per-run directory, as in PR0.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ("--harness", "--system", "--fcidump", "--config", "--diag-uhf", "--run-dir"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.harness)
    import inprocess_prefect  # noqa: F401  (in-process Prefect API, as in PR0)

    run = Path(a.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    solver_root = run / "solver_root"

    # 1. Blocks, with the port's own block creation.
    from uhf_sqd.create_blocks import create_blocks

    solver = {"hpc_target": "local", "root_dir": str(solver_root), "sbd_executable": a.diag_uhf}
    if a.config == "T0":
        solver.update(block=30, iteration=60, tolerance=1e-10)
    names = create_blocks(
        {"solver": solver, "quantum_sources": {"random-source": {"config": {"kind": "random"}}}}
    )

    # 2. Flow.
    from uhf_sqd.flow_params import FlowParameters
    from uhf_sqd.main import TELEMETRY_ARTIFACT_KEY, uhf_sqd_de

    params = {
        "fcidump": str(Path(a.fcidump).resolve()),
        "quantum_source": "random-source",
        "solver_block_ref": f"sbd_solver_job/{names['solver_block_name']}",
        "reference": {"cache_dir": str(run / "uhf_cache")},
    }
    if a.config == "D1":
        params.update(sqd_dim=400, n_recovery_steps=2, n_batches=2, de_params={"iterations": 2})
    fp = FlowParameters(**params)
    (run / "flow_parameters.json").write_text(fp.model_dump_json(indent=1))
    energy = uhf_sqd_de(parameters=fp)

    # 3. Outputs.
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey

    with get_client(sync_client=True) as client:
        arts = client.read_artifacts(
            artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=[TELEMETRY_ARTIFACT_KEY]))
        )
        blocks = {b.name: b.data for b in client.read_block_documents()}
    telemetry = [art.data for art in arts]

    storage = Path(os.environ["SBD_LOCAL_STORAGE_DIR"])
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

    # 4. Reference stage, the port's function (the cache written by the flow is read back).
    from qcsc_workflow_utility import uhf

    ep = uhf.compute_uhf_integrals_from_fcidump.fn(
        fcidump_file=str(Path(a.fcidump).resolve()), cache_dir=str(run / "uhf_cache")
    )
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
    cache = sorted((run / "uhf_cache").glob("*.npz"))
    s2 = e_tot_cache = None
    if cache:
        from pyscf.scf.uhf import spin_square

        with np.load(cache[0]) as c:
            mo_a, mo_b = c["mo_a"], c["mo_b"]
            s2 = float(spin_square((mo_a[:, :na], mo_b[:, :nb]), np.eye(mo_a.shape[0]))[0])
            e_tot_cache = float(c["e_tot"])
    solver_block = blocks.get(names["solver_block_name"], {})
    summary = {
        "system": a.system,
        "config": a.config,
        "flow_energy": energy,
        "ecore": float(ep.nuclear_repulsion_energy),
        "norb": ep.num_orbitals,
        "nelec": [na, nb],
        "E_HF_from_integrals_plus_ecore": float(e_hf + ep.nuclear_repulsion_energy),
        "E_UHF_cache": e_tot_cache,
        "reference_S2": s2,
        "sqd_data_files": [f.name for f in sqd_files],
        "solver_jobs": jobs,
        "telemetry": telemetry,
        "solver_block": {
            k: v
            for k, v in solver_block.items()
            if k
            in (
                "block",
                "iteration",
                "tolerance",
                "carryover_ratio",
                "carryover_type",
                "solver_mode",
                "integrals",
            )
        },
        "block_names": names,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("FLOW_ENERGY", repr(energy))


if __name__ == "__main__":
    main()
