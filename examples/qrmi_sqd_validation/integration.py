"""Local PR #42 sampling integration: pinned LUCJ -> QRMI/Slurm -> BitArray.

The original PR and its snapshot are unchanged. Runtime block loading and
workflow-host target acquisition are replaced by an injected Slurm runtime;
configuration recovery and Fulqrum remain outside this entrypoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from prefect import flow
from prefect.logging import get_run_logger
from pyscf import lib
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime
from qcsc_prefect.integrations.qiskit.qrmi.contract import positive_shots, resolve_shots
from qcsc_prefect.integrations.qiskit.qrmi.runtime import DEFAULT_TRANSPILER
from qiskit.primitives.containers import BitArray

from .pr42 import chem, load_pr42, prepare_sampler_circuit, source_metadata

LOGGER = logging.getLogger(__name__)
PR42_DEFAULT_SHOTS = 100_000
PR42_TRANSPILER = "examples.qrmi_sqd_validation.pr42:transpile_for_target"


async def sample_bitstrings(
    circuit_params,
    elec_props,
    *,
    runtime: QrmiRuntime,
    options: Mapping | None = None,
) -> BitArray:
    """Adapt the fixed PR's real-device sampling stage to an injected runtime.

    ``options`` replaces Prefect Variable lookup. Only ``params.shots`` is
    supported, with the PR's default of 100,000 when omitted. The runtime must
    be unconfigured for compilation: this adapter installs its pinned hooks on
    a copy, preserving the caller's resource, blocks, work directory and waits.
    No random fallback, result cache or retry is added here.
    """
    # Validate before circuit generation or any task/Slurm submission. Preserve
    # the PR's shot default while retaining the library's strict option checks.
    shots = resolve_shots(None, options)
    if options is None or "shots" not in options.get("params", {}):
        shots = PR42_DEFAULT_SHOTS
    sampler_options = {"params": {"shots": shots}}
    if (
        runtime.prepare is not None
        or runtime.transpiler != DEFAULT_TRANSPILER
        or runtime.transpile_config
    ):
        raise ValueError("Pass a runtime without compilation hooks/config; PR #42 owns them")
    pr42 = load_pr42()
    params = pr42.CircuitParameters.model_validate(circuit_params)
    circuit = pr42.create_ansatz_circuits(circuit_params=params, elec_props=elec_props)
    sampler = replace(
        runtime,
        prepare=partial(prepare_sampler_circuit, circuit_parameters=params.model_dump()),
        transpiler=PR42_TRANSPILER,
    )
    result = await sampler.sampler(sampler_pubs=[(circuit,)], options=sampler_options)
    meas_bits = result[0].data.meas
    if not params.use_reset_mitigation:
        return meas_bits

    # Match the pinned sample_bitstrings: select each meas shot by the same
    # test shot, retaining the original ordering, width and bit endianness.
    test_bits = result[0].data.test
    samples = meas_bits.get_bitstrings(test_bits.bitcount() == 0)
    if not samples:
        raise ValueError("Reset mitigation rejected all shots; no BitArray samples remain")
    bits = BitArray.from_samples(samples, num_bits=meas_bits.num_bits)
    LOGGER.info(
        "Reset mitigation: %s -> %s shots (retention %.6f)",
        meas_bits.num_shots,
        bits.num_shots,
        bits.num_shots / meas_bits.num_shots,
    )
    return bits


@flow(name="qrmi-pr42-h2-bitarray", retries=0, persist_result=False)
async def h2_bitarray_flow(
    *,
    shots: int = 100,
    work_dir: str = "/shared/qrmi-pr42-bitarray",
    reset_mitigation: bool = False,
) -> BitArray:
    """Executable H2 integration with ordered samples saved for the next stage."""
    shots = positive_shots(shots)
    run_dir = Path(work_dir).expanduser().resolve() / f"run_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    get_run_logger().info("PR #42 BitArray integration directory: %s", run_dir)
    params = load_pr42().CircuitParameters(
        use_reset_mitigation=reset_mitigation,
        sabre_layout_trials=4,
        sabre_swap_trials=2,
        sabre_max_iterations=2,
    )
    with lib.with_omp_threads(1):
        properties = chem.compute_molecular_integrals_from_geometry.fn(
            atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
        )
    runtime = QrmiRuntime(
        root_dir=run_dir / "quantum",
        command_block_name="poc-qrmi-command",
        execution_profile_block_name="poc-qrmi-execution",
        hpc_profile_block_name="poc-qrmi-hpc",
        resource="test_heron",
        scheduler="slurm",
    )
    bits = await sample_bitstrings(
        params, properties, runtime=runtime, options={"params": {"shots": shots}}
    )
    # One string per retained shot; counts alone would lose shot ordering.
    (run_dir / "bitstrings.txt").write_text(
        "\n".join(bits.get_bitstrings()) + "\n", encoding="utf-8"
    )
    summary = {
        "source": source_metadata(),
        "molecule": "H2",
        "basis": "sto-3g",
        "quantum_source": "real-device",
        "runtime": "qrmi",
        "scheduler": runtime.scheduler,
        "qiskit_integration_version": version("qcsc-prefect-qiskit"),
        "requested_shots": shots,
        "returned_shots": bits.num_shots,
        "num_bits": bits.num_bits,
        "reset_mitigation": reset_mitigation,
        "counts": bits.get_counts(),
        "work_dir": str(run_dir),
        "quantum_root_dir": str(runtime.root_dir),
    }
    (run_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return bits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--work-dir", default="/shared/qrmi-pr42-bitarray")
    parser.add_argument("--reset-mitigation", action="store_true")
    args = parser.parse_args(argv)
    bits = asyncio.run(h2_bitarray_flow(**vars(args)))
    print(
        json.dumps(
            {
                "num_bits": bits.num_bits,
                "num_shots": bits.num_shots,
                "counts": bits.get_counts(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
