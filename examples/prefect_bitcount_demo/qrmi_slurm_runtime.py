"""Compatibility facade retaining the validated example's block names and API."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

from qcsc_prefect.integrations.qiskit.qrmi import QrmiSlurmRuntime as LibraryRuntime


class QrmiSlurmRuntime(LibraryRuntime):
    """Keep PoC settings in the example; execution lives in the library."""

    def __init__(
        self,
        root_dir: Path,
        transpile_strategy: Literal["generic", "sqd-pr42"] = "generic",
        sqd_circuit_parameters: dict | None = None,
    ):
        if transpile_strategy not in {"generic", "sqd-pr42"}:
            raise ValueError(f"Unknown transpile strategy: {transpile_strategy}")
        if transpile_strategy == "generic" and sqd_circuit_parameters is not None:
            raise ValueError("sqd_circuit_parameters requires the sqd-pr42 strategy")
        self.transpile_strategy = transpile_strategy
        self.sqd_circuit_parameters = sqd_circuit_parameters
        kwargs = {}
        if transpile_strategy == "sqd-pr42":
            from examples.qrmi_sqd_validation.pr42 import prepare_sampler_circuit

            kwargs = {
                "prepare": partial(
                    prepare_sampler_circuit, circuit_parameters=sqd_circuit_parameters or {}
                ),
                "transpiler": "examples.qrmi_sqd_validation.pr42:transpile_for_target",
            }
        super().__init__(
            root_dir=root_dir,
            command_block_name="poc-qrmi-command",
            execution_profile_block_name="poc-qrmi-execution",
            hpc_profile_block_name="poc-qrmi-hpc",
            resource="test_heron",
            **kwargs,
        )
