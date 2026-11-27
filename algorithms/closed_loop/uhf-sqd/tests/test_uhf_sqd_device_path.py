"""The device path of Step 9: a QRMI source with scheduler="iqp", through the real sampling code.

The flow body runs on the NH2 doublet, without a Prefect server, as in ``test_uhf_sqd_flow.py``.
The source Block, the QRMI Runtime Block, ``QrmiSource``, ``QrmiRuntime`` and the UHF-SQD compile
hooks are the real ones. Only the ``qrmi`` module is a fake: it hands out the Target of
qiskit-ibm-runtime's fake heavy-hex backend and returns random ``meas`` bits and a ``test``
register with some ones. The solver is a stand-in. Nothing here reaches IBM Quantum.
"""

from __future__ import annotations

import json
import re
import socket
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
from qcsc_prefect.integrations.qiskit.sources import QuantumSourceBlock, QuantumSourceConfig
from qcsc_prefect_sbd import SBDResult, SBDSolverJob
from qcsc_workflow_utility import uhf
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.providers import JobStatus
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from uhf_sqd import main, sqd
from uhf_sqd.flow_params import FlowParameters

NH2 = str(Path(__file__).resolve().parents[1] / "data/fcidump_NH2_STO3G.txt")
NA, NB, NORB = 5, 4, 7
SHOTS = 600
REGISTER = re.compile(r"^bit\[(\d+)\] (\w+);$", re.MULTILINE)


class FakeIqp:
    """The QRMI calls of ``sample_on_iqp``: from_config, get_backend, task_start, the job."""

    def __init__(self):
        self.target = FakeMarrakesh().target
        self.rng = np.random.default_rng(5)
        self.resources = []
        self.jobs = []

    def result(self, job):
        data = {}
        for name, width in job["registers"].items():
            # About one shot in three has a 1 in the reset test, so mitigation drops shots.
            p_one = 0.5 if name == "meas" else 0.03
            bits = self.rng.random((job["shots"], width)) < p_one
            data[name] = BitArray.from_bool_array(bits, order="little")
        return PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})

    def modules(self):
        fake = self
        qrmi = ModuleType("qrmi")

        class ResourceType(Enum):
            IBMQuantumComputeService = "ibm-quantum-compute-service"

        class Payload:
            @staticmethod
            def QiskitPrimitive(*, input, program_id):  # noqa: N802 (QRMI's name)
                return SimpleNamespace(input=input, program_id=program_id)

        class QuantumResource:
            @staticmethod
            def from_config(resource_id, resource_type, config):
                resource = QuantumResource()
                resource.id, resource.config = resource_id, config
                fake.resources.append(resource)
                return resource

            def task_start(self, payload):
                parameters = json.loads(payload.input)
                (pub,) = parameters["pubs"]
                fake.jobs.append(
                    {
                        "id": f"job-{len(fake.jobs)}",
                        "qasm": pub[0],
                        "registers": {name: int(w) for w, name in REGISTER.findall(pub[0])},
                        "shots": parameters["shots"],
                    }
                )
                return fake.jobs[-1]["id"]

        class RuntimeJobV2:
            def __init__(self, resource, job_id, *, delete_job=False):
                self.job = next(job for job in fake.jobs if job["id"] == job_id)
                self.job["delete_job"] = delete_job

            def status(self):
                return JobStatus.DONE

            def result(self):
                return fake.result(self.job)

            def cancel(self):
                self.job["cancelled"] = True

        qrmi.QuantumResource, qrmi.ResourceType, qrmi.Payload = (
            QuantumResource,
            ResourceType,
            Payload,
        )
        primitives = ModuleType("qrmi.primitives")
        primitives.RuntimeJobV2 = RuntimeJobV2
        ibm = ModuleType("qrmi.primitives.ibm")
        ibm.get_backend = lambda resource: SimpleNamespace(target=fake.target)
        return {"qrmi": qrmi, "qrmi.primitives": primitives, "qrmi.primitives.ibm": ibm}


class Done:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class DoneList(list):
    def result(self):
        return [future.result() for future in self]


@pytest.fixture
def device_flow(monkeypatch, tmp_path):
    def reject_network(*args, **kwargs):
        pytest.fail("the device-path test must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)

    fake = FakeIqp()
    for name, module in fake.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    # The block of Step 9: scheduler="iqp", the instance and backend that qcsc-qrmi login saved.
    runtime_block = QrmiRuntimeBlock(
        scheduler="iqp",
        resource="ibm_marrakesh",
        instance="main (premium, us-east)",
        instances={"main (premium, us-east)": "crn:v1:test::"},
    )
    monkeypatch.setenv("QRMI_CONFIG", str(tmp_path / "missing-qrmi-config.json"))
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "test-api-key")
    loaded = []

    def load_runtime_block(cls, name, **kwargs):
        loaded.append(name)
        return runtime_block

    monkeypatch.setattr(QrmiRuntimeBlock, "load", classmethod(load_runtime_block))

    def load_source_block(cls, name, **kwargs):
        loaded.append(name)
        return QuantumSourceBlock(
            config=QuantumSourceConfig(kind="qrmi", block_name="uhf-sqd-iqp", tags=("res: qpu",))
        )

    monkeypatch.setattr(QuantumSourceBlock, "load", classmethod(load_source_block))
    monkeypatch.setattr(iqp_mod, "sample_on_iqp", iqp_mod.sample_on_iqp.fn)

    logger = SimpleNamespace(
        **{k: (lambda *a, **kw: None) for k in ("info", "debug", "exception", "warning", "error")}
    )
    for module in (main, sqd, uhf, iqp_mod):
        monkeypatch.setattr(module, "get_run_logger", lambda: logger)
    telemetry = []
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
    pools = []

    def save_ndarray(file_prefix, **arrays):
        # The sample pool, and the stored arrays (sqd_data) of each recovery pass.
        if file_prefix.startswith("raw_samples_"):
            pools.append(arrays)
        return f"file://{tmp_path}/{file_prefix}.npz"

    monkeypatch.setattr(sqd, "save_ndarray", save_ndarray)
    # The random path never builds a circuit; this one does.
    names = ("recover_configurations", "postselect_bitstrings", "subsample_open_shell")
    for name in (*names, "create_lucj_circuit"):
        monkeypatch.setattr(sqd, name, getattr(sqd, name).fn)

    solves = []

    class Solver:
        solver_mode = "cpu"
        integrals = "unrestricted"

        async def run(self, **kwargs):
            solves.append(kwargs)
            return SBDResult(
                energy=-54.0,
                orbital_occupancies=(np.full(NORB, NA / NORB), np.full(NORB, NB / NORB)),
                carryover_bitstrings=np.zeros((0, NORB), dtype=bool),
                carryover_bitstrings_b=np.zeros((0, NORB), dtype=bool),
            )

    monkeypatch.setattr(SBDSolverJob, "load", classmethod(lambda c, name: Solver()))

    def run():
        # The smoke-test settings of Step 9: one walker, one trial, few shots.
        parameters = FlowParameters(
            fcidump=NH2,
            quantum_source="uhf-sqd-qpu",
            sampler={"shots": SHOTS},
            de_params={"num_walkers": 1, "iterations": 1},
            circ_params={"optimization_level": 1, "sabre_layout_trials": 2},
            reference={"cache_dir": None},
        )
        return main.uhf_sqd_de.fn(parameters=parameters)

    return SimpleNamespace(
        run=run, fake=fake, loaded=loaded, telemetry=telemetry, pools=pools, solves=solves
    )


def test_one_walker_samples_once_on_the_qpu_and_solves(device_flow):
    energy = device_flow.run()

    assert np.isfinite(energy)
    assert device_flow.loaded == ["uhf-sqd-qpu", "uhf-sqd-iqp"]
    # One IQP job, with this host's API key and the instance's CRN, and no session.
    (resource,) = device_flow.fake.resources
    assert resource.id == "ibm_marrakesh"
    assert resource.config["QRMI_IBM_QCS_IAM_APIKEY"] == "test-api-key"
    assert resource.config["QRMI_IBM_QCS_SERVICE_CRN"] == "crn:v1:test::"
    assert "QRMI_IBM_QCS_SESSION_MODE" not in resource.config
    (job,) = device_flow.fake.jobs
    assert job["shots"] == SHOTS
    assert job["registers"] == {"test": 2 * NORB, "meas": 2 * NORB}
    assert job["delete_job"] is False and "cancelled" not in job
    # The circuit was compiled for the backend: an ISA circuit on its qubits and gates.
    assert f"qubit[{FakeMarrakesh().num_qubits}]" in job["qasm"] or "$" in job["qasm"]
    assert "cz " in job["qasm"] and "xx_plus_yy" not in job["qasm"]

    (row,) = device_flow.telemetry
    passed = row["reset_test_passed_shots"]
    assert 0 < passed < SHOTS
    assert row["reset_mitigation_fallback"] is False
    assert row["shot_retention_rate"] == pytest.approx(passed / SHOTS)
    assert row["raw_samples_path"].endswith("raw_samples_t00_w0.npz")
    (pool,) = device_flow.pools
    assert pool["num_bits"].tolist() == [2 * NORB]
    assert pool["num_shots"].tolist() == [passed]
    # The kept shots reached the solver, each spin with its own determinants.
    assert device_flow.solves
    alpha, beta = device_flow.solves[0]["ci_strings"]
    assert len(alpha) > 1 and len(beta) > 1
