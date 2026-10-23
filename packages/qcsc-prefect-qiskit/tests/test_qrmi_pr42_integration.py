"""Exercise SQD QPY, worker, QASM and CLI-result boundaries without Slurm/QPU."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import socket
import sys
from dataclasses import replace
from functools import partial
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("ffsim")
pytest.importorskip("pyscf")
pytest.importorskip("pydantic_numpy")
pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("qiskit_qasm3_import")
pytest.importorskip("qcsc_workflow_utility")

from pyscf import lib  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi import worker as node_mod  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi.contract import decode_result  # noqa: E402
from qiskit import qpy  # noqa: E402
from qiskit.primitives import PrimitiveResult, SamplerPubResult  # noqa: E402
from qiskit.primitives.containers import BitArray, DataBin  # noqa: E402
from qiskit.providers.fake_provider import GenericBackendV2  # noqa: E402

# Console-script pytest does not add the checkout root to sys.path.
# Keep the source-only application path local to its imports.
with patch.object(sys, "path", [str(Path(__file__).resolve().parents[3]), *sys.path]):
    integration = import_module("examples.qrmi_sqd_validation.integration")
    pr42_mod = import_module("examples.qrmi_sqd_validation.pr42")
FIXTURE_DIR = pr42_mod.FIXTURE_DIR
chem = pr42_mod.chem
load_pr42 = pr42_mod.load_pr42

LOGGER = SimpleNamespace(info=lambda *args, **kwargs: None)


@pytest.fixture(scope="module", autouse=True)
def block_network():
    def fail(*args, **kwargs):
        pytest.fail("The PR #42 integration tests must not open network connections")

    with (
        patch.object(socket.socket, "connect", side_effect=fail),
        patch.object(socket.socket, "connect_ex", side_effect=fail),
        patch.object(socket, "create_connection", side_effect=fail),
    ):
        yield


@pytest.fixture(scope="module")
def h2(block_network):
    with lib.with_omp_threads(1), patch.object(chem, "get_run_logger", return_value=LOGGER):
        return chem.compute_molecular_integrals_from_geometry.fn(
            atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
        )


def parameters(reset):
    return load_pr42().CircuitParameters(
        use_reset_mitigation=reset,
        sabre_layout_trials=4,
        sabre_swap_trials=2,
        sabre_max_iterations=2,
    )


@pytest.fixture
def fake_qrmi_node(tmp_path, monkeypatch):
    monkeypatch.setenv("test_heron_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.setenv("test_heron_QRMI_JOB_ACQUISITION_TOKEN", "test-token")
    resource = SimpleNamespace(resource_id=lambda: "test_heron")
    qrmi = ModuleType("qrmi")
    qrmi.QRMIService = lambda: SimpleNamespace(resources=lambda: [resource])
    ibm = ModuleType("qrmi.primitives.ibm")
    monkeypatch.setitem(sys.modules, "qrmi", qrmi)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", ModuleType("qrmi.primitives"))
    monkeypatch.setitem(sys.modules, "qrmi.primitives.ibm", ibm)

    runner = tmp_path / "fake_task_runner"
    runner.write_text(
        f"#!{sys.executable}\n"
        # This executable performs no service calls: it parses the real QASM
        # payload and returns encoded sample containers to test the CLI boundary.
        """import json, socket, sys
def offline(*args, **kwargs):
    raise RuntimeError('The fake CLI must not open network connections')
socket.socket.connect = offline
socket.socket.connect_ex = offline
socket.create_connection = offline
from pathlib import Path
from qiskit import qasm3
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_ibm_runtime import RuntimeEncoder
resource, source, destination = sys.argv[1:]
assert resource == 'test_heron'
payload = json.loads(Path(source).read_text())
assert payload['program_id'] == 'sampler'
params = payload['parameters']
assert params['support_qiskit'] is True
qasm, values, shots = params['pubs'][0]
assert values is None and shots == params['shots']
circuit = qasm3.loads(qasm)
# Physical QASM can omit unused Target qubits; the ISA QPY keeps all six.
assert 4 <= circuit.num_qubits <= 6
assert {r.name: len(r) for r in circuit.cregs}.get('meas') == 4
data = {}
overrides_path = Path('fake_readouts.json')
overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
for reg in circuit.cregs:
    assert len(reg) == 4
    samples = overrides.get(
        reg.name, ['0000' if reg.name == 'test' else '0101' for _ in range(shots)]
    )
    assert len(samples) == shots
    data[reg.name] = BitArray.from_samples(samples, num_bits=len(reg))
result = PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})
Path(destination).write_text(json.dumps(result, cls=RuntimeEncoder))
""",
        encoding="utf-8",
    )
    runner.chmod(0o700)
    execute = node_mod.execute_task_runner
    monkeypatch.setattr(
        node_mod,
        "execute_task_runner",
        lambda isa, shots, _, resource: execute(isa, shots, runner, resource),
    )

    def reject_generic(*args, **kwargs):
        pytest.fail("SQD strategy must call the PR-specific transpiler")

    monkeypatch.setattr(node_mod, "generate_preset_pass_manager", reject_generic)
    monkeypatch.setattr(runtime_mod, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    return ibm


@pytest.mark.parametrize("with_rzz", [False, True], ids=["cz", "rzz"])
@pytest.mark.parametrize("reset", [False, True], ids=["meas", "meas-test"])
def test_sampler_qpy_worker_qasm_cli_handoff(
    tmp_path, monkeypatch, h2, fake_qrmi_node, reset, with_rzz
):
    gates = ["rz", "sx", "x", "cz"] + (["rzz"] if with_rzz else [])
    edges = [(i, i + 1) for i in range(5)] + [(i + 1, i) for i in range(5)]
    target = GenericBackendV2(num_qubits=6, basis_gates=gates, coupling_map=edges, seed=42).target
    fake_qrmi_node.get_target = lambda resource: target
    params = parameters(reset)
    original = load_pr42().create_ansatz_circuits(params, h2)

    async def fake_submit(**kwargs):
        assert kwargs["hpc_profile_block_name"] == "poc-qrmi-hpc"
        assert kwargs["user_args"] == ["--request", "sampler_request.json"]
        work_dir = kwargs["work_dir"]
        monkeypatch.chdir(work_dir)
        monkeypatch.setattr(sys, "argv", ["qrmi_task_runner.py", *kwargs["user_args"]])
        node_mod.main()
        return SimpleNamespace(job_id="12", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", fake_submit)
    runtime = replace(
        make_library_runtime(tmp_path / "quantum"),
        prepare=partial(pr42_mod.prepare_sampler_circuit, circuit_parameters=params.model_dump()),
        transpiler=integration.PR42_TRANSPILER,
    )
    result = asyncio.run(runtime.sampler([(original,)], shots=4))
    work_dir = Path(result.metadata["qcsc_slurm"]["work_dir"])
    config = json.loads((work_dir / "transpile_config.json").read_text())
    report = json.loads((work_dir / "transpile_report.json").read_text())
    assert config["original_ops"] == dict(original.count_ops())
    assert report["input_ops"] != report["original_ops"]
    assert report["strategy"] == "sqd-pr42"
    assert report["mode"] == "batch"
    expected_registers = {"meas": 4, **({"test": 4} if reset else {})}
    assert report["registers"] == expected_registers
    assert result.metadata["qcsc_slurm"]["transpiler"] == integration.PR42_TRANSPILER
    assert result[0].data.meas.get_counts() == {"0101": 4}
    if reset:
        assert result[0].data.test.get_counts() == {"0000": 4}
    with (work_dir / "isa.qpy").open("rb") as stream:
        (isa,) = qpy.load(stream)
    assert isa.num_qubits == 6
    assert {reg.name: len(reg) for reg in isa.cregs} == expected_registers


def test_worker_rejects_different_pr_source_before_cli(tmp_path, monkeypatch, h2, fake_qrmi_node):
    fake_qrmi_node.get_target = lambda resource: (
        GenericBackendV2(num_qubits=6, basis_gates=["rz", "sx", "x", "cz"]).target
    )
    params = parameters(False)
    circuit = load_pr42().create_ansatz_circuits(params, h2)

    async def fake_submit(**kwargs):
        work_dir = kwargs["work_dir"]
        config_path = work_dir / "transpile_config.json"
        config = json.loads(config_path.read_text())
        config["source_commit"] = "another-commit"
        config_path.write_text(json.dumps(config))
        monkeypatch.chdir(work_dir)
        monkeypatch.setattr(sys, "argv", ["qrmi_task_runner.py", *kwargs["user_args"]])
        node_mod.main()

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", fake_submit)
    runtime = replace(
        make_library_runtime(tmp_path / "quantum"),
        prepare=partial(pr42_mod.prepare_sampler_circuit, circuit_parameters=params.model_dump()),
        transpiler=integration.PR42_TRANSPILER,
    )
    with pytest.raises(ValueError, match="source commit mismatch"):
        asyncio.run(runtime.sampler([(circuit,)], shots=4))
    assert not list(tmp_path.glob("quantum/*/sampler_input.json"))


def make_library_runtime(root):
    return runtime_mod.QrmiRuntime(
        root_dir=root,
        command_block_name="poc-qrmi-command",
        execution_profile_block_name="poc-qrmi-execution",
        hpc_profile_block_name="poc-qrmi-hpc",
        resource="test_heron",
        scheduler="slurm",
    )


async def reference_bitstrings(params, properties, result, options):
    """Execute the unmodified pinned sampling function as a postprocessing oracle.

    Its direct Runtime/target/compiler boundary is stubbed here only. The tested
    local integration runs the real QPY, pinned compiler and CLI path separately.
    """
    from typing import Any, Literal

    provenance = json.loads((FIXTURE_DIR / "source.json").read_text())
    source = (FIXTURE_DIR / "sqd_fulqrum_main.py.txt").read_bytes()
    assert hashlib.sha256(source).hexdigest() == provenance["sha256"]
    tree = ast.parse(source)
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_resolve_shots", "sample_bitstrings"}
    ]
    assert len(definitions) == 2
    circuit = object()

    class ReferenceRuntime:
        async def get_target(self):
            return None

        async def sampler(self, *, sampler_pubs, options):
            assert sampler_pubs == [(circuit,)]
            return result

    async def load(_):
        return ReferenceRuntime()

    async def get_options(_, *, default):
        return options if options is not None else default

    namespace = {
        **load_pr42().__dict__,
        "Any": Any,
        "Literal": Literal,
        "BitArray": BitArray,
        "task": lambda **kwargs: lambda fn: fn,
        "Variable": SimpleNamespace(get=get_options),
        "QuantumRuntime": SimpleNamespace(load=load),
        "create_ansatz_circuits": lambda **kwargs: circuit,
        "transpile_circuit": lambda **kwargs: circuit,
        "get_run_logger": lambda: LOGGER,
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), "pinned-sampling", "exec"), namespace
    )
    return await namespace["sample_bitstrings"](
        circuit_params=params,
        elec_props=properties,
        runner_name="reference-runtime",
        option_name="reference-options",
        quantum_source="real-device",
        random_seed=24,
    )


@pytest.mark.parametrize("with_rzz", [False, True], ids=["cz", "rzz"])
@pytest.mark.parametrize("reset", [False, True], ids=["meas", "meas-test"])
def test_pr42_integration_returns_reference_bitarray_via_qrmi(
    tmp_path, monkeypatch, h2, fake_qrmi_node, reset, with_rzz
):
    gates = ["rz", "sx", "x", "cz"] + (["rzz"] if with_rzz else [])
    edges = [(i, i + 1) for i in range(5)] + [(i + 1, i) for i in range(5)]
    fake_qrmi_node.get_target = lambda _: (
        GenericBackendV2(num_qubits=6, basis_gates=gates, coupling_map=edges, seed=42).target
    )
    params = parameters(reset)
    # Asymmetric bitstrings expose endianness changes; nonzero test shots
    # expose accidentally selecting the wrong shot, or never filtering at all.
    meas = ["0101", "1010", "0001", "1111"] * 25
    test = ["0000", "0001", "0010", "0000"] * 25
    submissions = []

    async def submit(**kwargs):
        submissions.append(kwargs)
        work_dir = kwargs["work_dir"]
        (work_dir / "fake_readouts.json").write_text(json.dumps({"meas": meas, "test": test}))
        monkeypatch.chdir(work_dir)
        node_mod.main(kwargs["user_args"])
        return SimpleNamespace(job_id="16", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    runtime = make_library_runtime(tmp_path / "quantum")
    runtime.watch_poll_interval = 0.5
    runtime.timeout_seconds = 120
    options = {"params": {"shots": 100}}
    bits = asyncio.run(integration.sample_bitstrings(params, h2, runtime=runtime, options=options))
    assert len(submissions) == 1
    assert submissions[0]["watch_poll_interval"] == 0.5
    assert submissions[0]["timeout_seconds"] == 120
    assert submissions[0]["command_block_name"] == runtime.command_block_name
    assert submissions[0]["execution_profile_block_name"] == runtime.execution_profile_block_name
    assert submissions[0]["hpc_profile_block_name"] == runtime.hpc_profile_block_name
    assert runtime.scheduler == "slurm"
    job_dir = submissions[0]["work_dir"]
    config = json.loads((job_dir / "transpile_config.json").read_text())
    request = json.loads((job_dir / "sampler_request.json").read_text())
    assert request["shots"] == 100
    assert request["transpiler"] == integration.PR42_TRANSPILER
    assert config["source_commit"] == "2d435b19af4648cbbc5fbc6a27ff258c93817d1a"
    assert config["transport"] == "ffsim-pre-init-decomposed"
    assert config["circuit_parameters"] == params.model_dump()
    assert runtime.prepare is None
    assert runtime.transpiler == runtime_mod.DEFAULT_TRANSPILER
    assert options == {"params": {"shots": 100}}

    registers = {"meas": 4, **({"test": 4} if reset else {})}
    raw_result = decode_result(job_dir / "result.json", 100, registers)
    reference = asyncio.run(reference_bitstrings(params, h2, raw_result, options))
    assert isinstance(bits, BitArray)
    assert bits.shape == ()
    assert bits.num_bits == reference.num_bits == 4
    assert bits.num_shots == reference.num_shots == (50 if reset else 100)
    expected = ["0101", "1111"] * 25 if reset else meas
    assert bits.get_bitstrings() == reference.get_bitstrings() == expected
    assert bits.get_counts() == reference.get_counts()


@pytest.mark.parametrize("options", [None, {}, {"params": {}}])
def test_pr42_integration_preserves_pr_default_shots(tmp_path, monkeypatch, h2, options):
    async def sampler(self, *, sampler_pubs, options):
        assert options == {"params": {"shots": 100_000}}
        return PrimitiveResult(
            [SamplerPubResult(DataBin(meas=BitArray.from_samples(["0101"] * 100_000)), metadata={})]
        )

    monkeypatch.setattr(runtime_mod.QrmiRuntime, "sampler", sampler)
    bits = asyncio.run(
        integration.sample_bitstrings(
            parameters(False), h2, runtime=make_library_runtime(tmp_path), options=options
        )
    )
    assert bits.num_shots == 100_000


@pytest.mark.parametrize(
    "options",
    [
        {"params": {"shots": True}},
        {"params": {"shots": 0}},
        {"params": {"shots": 100}, "resilience_level": 1},
    ],
)
def test_pr42_integration_rejects_options_before_circuit_or_submission(
    tmp_path, monkeypatch, h2, options
):
    params = parameters(False)

    def reject_load():
        pytest.fail("Invalid options must fail before circuit generation")

    monkeypatch.setattr(integration, "load_pr42", reject_load)
    with pytest.raises(ValueError):
        asyncio.run(
            integration.sample_bitstrings(
                params, h2, runtime=make_library_runtime(tmp_path / "jobs"), options=options
            )
        )
    assert not (tmp_path / "jobs").exists()


def test_pr42_integration_does_not_return_data_when_submission_fails(
    tmp_path, monkeypatch, h2, fake_qrmi_node
):
    async def fail(**kwargs):
        return SimpleNamespace(job_id="17", state="FAILED", exit_status=1)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", fail)
    with pytest.raises(RuntimeError, match="job 17 failed"):
        asyncio.run(
            integration.sample_bitstrings(
                parameters(False),
                h2,
                runtime=make_library_runtime(tmp_path),
                options={"params": {"shots": 100}},
            )
        )


def test_pr42_integration_reports_zero_retained_shots(tmp_path, monkeypatch, h2):
    raw_result = PrimitiveResult(
        [
            SamplerPubResult(
                DataBin(
                    meas=BitArray.from_samples(["0101", "1010"]),
                    test=BitArray.from_samples(["0001", "1000"]),
                ),
                metadata={},
            )
        ]
    )

    async def sampler(self, **kwargs):
        return raw_result

    monkeypatch.setattr(runtime_mod.QrmiRuntime, "sampler", sampler)
    params = parameters(True)
    options = {"params": {"shots": 2}}
    with pytest.raises(ValueError, match="rejected all shots"):
        asyncio.run(
            integration.sample_bitstrings(
                params, h2, runtime=make_library_runtime(tmp_path), options=options
            )
        )
    # The pinned implementation also cannot construct an empty BitArray.
    with pytest.raises(ValueError):
        asyncio.run(reference_bitstrings(params, h2, raw_result, options))


@pytest.mark.parametrize("reset", [False, True], ids=["meas", "meas-test"])
def test_h2_bitarray_entrypoint_runs_library_and_saves_ordered_samples(
    tmp_path, monkeypatch, h2, fake_qrmi_node, reset
):
    fake_qrmi_node.get_target = lambda _: (
        GenericBackendV2(num_qubits=6, basis_gates=["rz", "sx", "x", "cz"], seed=42).target
    )
    monkeypatch.setattr(chem.compute_molecular_integrals_from_geometry, "fn", lambda **kwargs: h2)
    monkeypatch.setattr(integration, "get_run_logger", lambda: LOGGER)

    async def submit(**kwargs):
        monkeypatch.chdir(kwargs["work_dir"])
        node_mod.main(kwargs["user_args"])
        return SimpleNamespace(job_id="18", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    bits = asyncio.run(
        integration.h2_bitarray_flow.fn(shots=100, work_dir=str(tmp_path), reset_mitigation=reset)
    )
    (run_dir,) = tmp_path.glob("run_*")
    summary = json.loads((run_dir / "sampling_summary.json").read_text())
    assert summary["requested_shots"] == summary["returned_shots"] == 100
    assert summary["num_bits"] == 4
    assert summary["reset_mitigation"] is reset
    assert summary["runtime"] == "qrmi"
    assert summary["scheduler"] == "slurm"
    assert summary["qiskit_integration_version"] == "0.3.0"
    assert summary["quantum_source"] == "real-device"
    assert summary["source"]["commit"] == "2d435b19af4648cbbc5fbc6a27ff258c93817d1a"
    assert summary["counts"] == {"0101": 100}
    samples = (run_dir / "bitstrings.txt").read_text().splitlines()
    restored = BitArray.from_samples(samples, num_bits=summary["num_bits"])
    assert restored.get_bitstrings() == bits.get_bitstrings()


def test_pinned_hook_imports_from_job_directory_in_fresh_interpreter(tmp_path):
    import os
    import subprocess

    program = """
import hashlib, importlib.abc, sys
class RejectComputeServices(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'qrmi', 'qcsc_prefect_executor'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectComputeServices())
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime
from examples.qrmi_sqd_validation.pr42 import FIXTURE_DIR, load_pr42, source_metadata
from examples.qrmi_sqd_validation.pr42 import transpile_for_target
metadata = source_metadata()
assert metadata['commit'] == '2d435b19af4648cbbc5fbc6a27ff258c93817d1a'
assert hashlib.sha256((FIXTURE_DIR / 'sqd_fulqrum_main.py.txt').read_bytes()).hexdigest() \
    == metadata['sha256']
assert load_pr42().CircuitParameters is not None
assert callable(transpile_for_target)
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
