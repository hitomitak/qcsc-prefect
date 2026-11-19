"""The UHF-SQD flow body runs to the end with a random source and a fake solver.

It runs without a Prefect server: its tasks are called directly, walkers run one after another,
and artifacts are recorded instead of uploaded. The reference is the real BS-UHF reference of the
NH2 doublet FCIDUMP (7 orbitals, 5 + 4 electrons).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qcsc_prefect.integrations.qiskit.sources import QuantumSourceBlock, QuantumSourceConfig
from qcsc_prefect_sbd import SBDResult, SBDSolverJob
from qcsc_workflow_utility import uhf
from uhf_sqd import main, sqd
from uhf_sqd.flow_params import FlowParameters

NH2 = str(Path(__file__).resolve().parents[1] / "data/fcidump_NH2_STO3G.txt")
NA, NB, NORB = 5, 4, 7


class Done:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class DoneList(list):
    def result(self):
        return [future.result() for future in self]


@pytest.fixture
def flow(monkeypatch, tmp_path):
    monkeypatch.setattr(
        QuantumSourceBlock,
        "load",
        classmethod(
            lambda cls, name, **kwargs: QuantumSourceBlock(config=QuantumSourceConfig("random"))
        ),
    )
    logger = SimpleNamespace(
        **{k: (lambda *a, **kw: None) for k in ("info", "debug", "exception", "warning")}
    )
    telemetry = []
    for module in (main, sqd, uhf):
        monkeypatch.setattr(module, "get_run_logger", lambda: logger)
    monkeypatch.setattr(main, "create_table_artifact", lambda **kwargs: None)
    monkeypatch.setattr(
        main, "extend_table_artifact", lambda artifact_key, new_table: telemetry.extend(new_table)
    )
    monkeypatch.setattr(
        main, "compute_uhf_integrals_from_fcidump", uhf.compute_uhf_integrals_from_fcidump.fn
    )
    for name in (
        "differential_evolution_trial",
        "mutation_and_crossover",
        "selection",
        "initialize_ucj_parameters",
    ):
        monkeypatch.setattr(main, name, getattr(main, name).fn)
    monkeypatch.setattr(main, "PrefectFutureList", DoneList)
    monkeypatch.setattr(
        main, "walker_sqd", SimpleNamespace(submit=lambda **kw: Done(sqd.walker_sqd.fn(**kw)))
    )
    monkeypatch.setattr(sqd, "save_ndarray", lambda **arrays: f"file://{tmp_path}/data.npz")
    for name in ("recover_configurations", "postselect_bitstrings", "subsample_open_shell"):
        monkeypatch.setattr(sqd, name, getattr(sqd, name).fn)

    def run(solver_block_ref="sbd_solver_job/sbd-uhf-solver", iterations=2):
        parameters = FlowParameters(
            fcidump=NH2,
            sqd_dim=100,
            quantum_source="uhf-sqd-random",
            solver_block_ref=solver_block_ref,
            sampler={"shots": 2000},
            de_params={"num_walkers": 4, "iterations": iterations},
            reference={"cache_dir": None},
        )
        return main.uhf_sqd_de.fn(parameters=parameters)

    run.telemetry = telemetry
    return run


def fake_solver(monkeypatch, integrals="unrestricted"):
    inputs = []

    class Fake:
        solver_mode = "cpu"

        def __init__(self):
            self.integrals = integrals

        async def run(self, **kwargs):
            inputs.append(kwargs)
            a, b = kwargs["ci_strings"]
            return SBDResult(
                energy=-1.0 - 0.001 * len(a) * len(b),
                orbital_occupancies=(np.full(NORB, NA / NORB), np.full(NORB, NB / NORB)),
                # A non-Hartree-Fock determinant per spin, so that it reaches the next generation.
                carryover_bitstrings=np.array([[True] * 4 + [False, True, False]]),
                carryover_bitstrings_b=np.array([[True] * 3 + [False, True, False, False]]),
            )

    monkeypatch.setattr(SBDSolverJob, "load", classmethod(lambda c, name: Fake()))
    return inputs


def test_closed_loop_passes_per_spin_carryover_to_the_next_generation(flow, monkeypatch):
    inputs = fake_solver(monkeypatch)
    energy = flow()
    assert len(inputs) == 8  # four walkers in each of two generations
    # NH2's constant term (7.78 Eh) makes these total energies positive; the flow returns the
    # lowest of them, not the source's initial 0.0.
    energies = [row["energy"] for row in flow.telemetry]
    assert min(energies) > 0 and energy == min(energies)
    carried_a, carried_b = int("1111010", 2), int("1110100", 2)
    first, second = inputs[:4], inputs[4:]
    assert all(carried_a not in call["ci_strings"][0] for call in first)
    assert all(carried_a in call["ci_strings"][0] for call in second)
    assert all(carried_b in call["ci_strings"][1] for call in second)
    assert all(call["ci_strings"][0][0] == (1 << NA) - 1 for call in inputs)
    assert all(call["ci_strings"][1][0] == (1 << NB) - 1 for call in inputs)
    assert all(call["one_body_tensor_b"].shape == (NORB, NORB) for call in inputs)
    assert len(flow.telemetry) == 8
    assert {(row["norb"], tuple(row["nelec"])) for row in flow.telemetry} == {(NORB, (NA, NB))}


def test_closed_loop_refuses_a_restricted_solver_block_before_any_walker(flow, monkeypatch):
    inputs = fake_solver(monkeypatch, integrals="restricted")
    with pytest.raises(ValueError, match="integrals='unrestricted'"):
        flow()
    assert inputs == [] and flow.telemetry == []


def test_closed_loop_rejects_an_unknown_solver_before_any_walker(flow, monkeypatch):
    fake_solver(monkeypatch)
    with pytest.raises(ValueError, match="solver_block_ref"):
        flow("other_solver/x")
    assert flow.telemetry == []


def test_the_flow_uses_an_explicit_thread_pool_task_runner():
    from prefect.task_runners import ThreadPoolTaskRunner

    assert isinstance(main.uhf_sqd_de.task_runner, ThreadPoolTaskRunner)
    assert main.uhf_sqd_de.name == "uhf_sqd_de"
