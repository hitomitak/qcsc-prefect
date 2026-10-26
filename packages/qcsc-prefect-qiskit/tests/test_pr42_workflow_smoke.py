"""Reviewable source preservation and the original flow with a QRMI boundary."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ffsim")
pytest.importorskip("qcsc_workflow_utility")
pytest.importorskip("qcsc_prefect_fulqrum")

from prefect.artifacts import create_table_artifact
from prefect.testing.utilities import prefect_test_harness
from qcsc_prefect.integrations.qiskit.qrmi import runtime as quantum_runtime
from qcsc_prefect_executor import from_blocks
from qcsc_prefect_fulqrum import solver_job

# Reuse explicit fake quantum/native boundaries; all actual algorithms stay real.
from test_qrmi_end_to_end import boundaries as _boundaries
from test_qrmi_end_to_end import no_external_network  # noqa: F401

boundaries = _boundaries

smoke = importlib.import_module("examples.qrmi_sqd_validation.pr42_smoke")
proof_module = importlib.import_module("examples.qrmi_sqd_validation.pr42_source_proof")
workflow = importlib.import_module("examples.qrmi_sqd_validation.pr42_workflow")
bridge = importlib.import_module("examples.qrmi_sqd_validation.pr42_bridge")
DIRECTORY = Path(smoke.__file__).parent


def test_source_copy_preserves_calculation_and_parameters():
    proof = proof_module.verify_workflow_copy()
    assert {
        "sqd_2405_05068_fulqrum",
        "make_block_solver",
        "create_ansatz_circuits",
        "transpile_circuit",
        "Parameters",
        "SQDParameters",
    } <= set(proof["unchanged_definitions"])
    assert proof["changed_definition"] == "sample_bitstrings"
    assert proof["sampling_postprocessing_unchanged"] is True
    assert (
        workflow.run_configuration_recovery is bridge._original_recovery.run_configuration_recovery
    )
    from qcsc_prefect_fulqrum import FulqrumSolverJob

    assert workflow.FulqrumSolverJob is FulqrumSolverJob
    payload = json.loads((DIRECTORY / "pr42_smoke_parameters.json").read_text())
    parameters = workflow.Parameters.model_validate(payload["parameters"])
    assert parameters.sqd.subspace_dim == 1
    assert parameters.sqd.num_batches == parameters.sqd.max_iterations == 1
    assert parameters.circuit.sabre_layout_trials == parameters.circuit.sabre_swap_trials == 1
    assert payload["sampler_options"]["params"]["shots"] == 16


def test_displayed_patch_applies_to_the_pinned_source(tmp_path):
    target = tmp_path / "algorithms/sqd/sqd_fulqrum/main.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(smoke.FIXTURE_DIR / "sqd_fulqrum_main.py.txt", target)
    result = subprocess.run(
        ["git", "apply", str(DIRECTORY / "pr42_qrmi.patch")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == (DIRECTORY / "pr42_workflow.py").read_bytes()


def test_source_proof_rejects_changes_to_the_calculation(tmp_path, monkeypatch):
    source = (DIRECTORY / "pr42_workflow.py").read_text()
    source = source.replace(
        "range(0, elec_props.num_orbitals, 4)", "range(0, elec_props.num_orbitals, 2)"
    )
    (tmp_path / "pr42_workflow.py").write_text(source)
    monkeypatch.setattr(proof_module, "__file__", str(tmp_path / "pr42_source_proof.py"))
    with pytest.raises(ValueError, match="Calculation definition changed: create_ansatz_circuits"):
        proof_module.verify_workflow_copy()


def test_proof_cli_without_server_or_checkout_cwd(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "examples.qrmi_sqd_validation.pr42_smoke", "proof"],
        cwd=tmp_path,
        env={
            **{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            "PREFECT_API_URL": "http://127.0.0.1:1/api",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(json.loads(result.stdout)["unchanged_definitions"]) == 11


@pytest.mark.parametrize("failed_with_output", [False, True])
def test_original_flow_runs_and_does_not_require_fci(
    tmp_path, monkeypatch, boundaries, failed_with_output
):
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_API_KEY", "")
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

    async def load_solver(name):
        return boundaries.solver

    monkeypatch.setattr(workflow.FulqrumSolverJob, "load", load_solver)
    quantum = quantum_runtime.run_job_from_blocks
    classical = from_blocks.run_job_from_blocks

    async def recorded(submit, artifact_key, **kwargs):
        job = await submit(**kwargs)
        if failed_with_output and artifact_key != "qrmi-slurm-job":
            job = SimpleNamespace(job_id=job.job_id, state="FAILED", exit_status=1)
        row = {
            "job_id": job.job_id,
            "state": job.state,
            "exit_code": str(job.exit_status),
            "signal": "0",
        }
        await create_table_artifact(table=[list(row), list(row.values())], key=artifact_key)
        return job

    async def quantum_submit(**kwargs):
        return await recorded(quantum, "qrmi-slurm-job", **kwargs)

    async def classical_submit(**kwargs):
        return await recorded(classical, boundaries.solver.metrics_artifact_key, **kwargs)

    monkeypatch.setattr(quantum_runtime, "run_job_from_blocks", quantum_submit)
    # The original solver module imports its executor directly; retain that interface.
    monkeypatch.setattr(solver_job, "run_job_from_blocks", classical_submit)

    async def exercise():
        return await smoke.run_smoke(solver_block="test-solver", work_dir=str(tmp_path / "smoke"))

    with prefect_test_harness():
        if failed_with_output:
            with pytest.raises(ValueError, match="Scheduler job failed"):
                asyncio.run(exercise())
            (root,) = (tmp_path / "smoke").iterdir()
            assert json.loads((root / "failure.json").read_text())["phase"] == "workflow-check"
            assert not (root / "workflow_summary.json").exists()
            return
        summary = asyncio.run(exercise())
    assert summary["validation"] == "workflow-passed"
    assert summary["accuracy_checked"] is False
    assert summary["requested_shots"] == summary["retained_shots"] == 16
    assert boundaries.calls == ["quantum", "classical"]
    assert len(summary["scheduler_jobs"]) == 2
    assert summary["recovery_records"][0]["dimension"] == 1
    # The original flow succeeds even though the 1D result is not the full H2 FCI energy.
    assert abs(summary["total_energy"] - (-1.1372838344885023)) > 1e-3
    assert (Path(summary["work_dir"]) / "pr42_qrmi.patch").read_bytes() == (
        DIRECTORY / "pr42_qrmi.patch"
    ).read_bytes()
