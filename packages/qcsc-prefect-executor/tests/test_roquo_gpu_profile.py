"""Job scripts for ROQUO profiles: examples/roquo_prefect_gpu_hello_demo on the Slurm executor.

The Blocks come from the example's own config and ``build_blocks``, and the script is built by
the same path as a real submission: ``run_job_from_blocks`` up to ``run_slurm_job``, then the
Slurm template.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import stat
import subprocess
import tomllib
from pathlib import Path

import pytest
from qcsc_prefect_adapters.slurm import builder as slurm_builder
from qcsc_prefect_executor import from_blocks as mod

DEMO_DIR = Path(__file__).resolve().parents[3] / "examples" / "roquo_prefect_gpu_hello_demo"


def _load_demo_module(name: str):
    spec = importlib.util.spec_from_file_location(f"roquo_demo_{name}", DEMO_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


create_blocks = _load_demo_module("create_blocks")


def _example_config(**changes) -> dict:
    with (DEMO_DIR / "roquo_gpu.example.toml").open("rb") as f:
        config = tomllib.load(f)
    config["account"] = "rq-test"
    config.update(changes)
    return config


def _render(monkeypatch, tmp_path: Path, config: dict, user_args=("0",)) -> str:
    blocks = create_blocks.build_blocks(config)
    by_type = {type(block).__name__: block for block in blocks.values()}

    def api(block):
        async def load(_name: str):
            return block

        return type("_API", (), {"load": staticmethod(load)})

    monkeypatch.setattr(mod, "CommandBlock", api(by_type["CommandBlock"]))
    monkeypatch.setattr(mod, "ExecutionProfileBlock", api(by_type["ExecutionProfileBlock"]))
    monkeypatch.setattr(mod, "HPCProfileBlock", api(by_type["HPCProfileBlock"]))
    captured = {}

    async def fake_run_slurm_job(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mod, "run_slurm_job", fake_run_slurm_job)
    asyncio.run(
        mod.run_job_from_blocks(
            command_block_name="cmd-roquo-gpu-hello",
            execution_profile_block_name="exec-roquo-gpu-1",
            hpc_profile_block_name="hpc-roquo-gpu",
            work_dir=tmp_path,
            script_filename="roquo_gpu_hello.slurm",
            user_args=list(user_args),
        )
    )
    assert captured["script_filename"] == "roquo_gpu_hello.slurm"
    return slurm_builder.render_script(
        work_dir=captured["work_dir"], exec_profile=captured["exec_profile"], req=captured["req"]
    )


def _directives(script: str) -> list[str]:
    return [line for line in script.splitlines() if line.startswith("#SBATCH")]


def test_example_config_gives_one_gpu_on_the_roquo_partition(monkeypatch, tmp_path):
    script = _render(monkeypatch, tmp_path, _example_config(), user_args=["3"])
    work_dir = tmp_path.resolve()
    assert _directives(script) == [
        "#SBATCH --partition=roquo",
        "#SBATCH --account=rq-test",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --time=00:05:00",
        f"#SBATCH --output={work_dir}/output.out",
        f"#SBATCH --error={work_dir}/output.err",
    ]
    assert "module load" not in script
    assert f'QCSC_PREFECT_EXECUTABLE="{DEMO_DIR / "hello_gpu.sh"}"' in script
    assert script.rstrip().endswith('srun "${QCSC_PREFECT_EXECUTABLE}" 3')


def test_account_reservation_modules_and_directives_reach_the_script(monkeypatch, tmp_path):
    config = _example_config(
        account="rq-proj",
        partition="roquo-debug",
        reservation="rq-resv",
        modules=["cuda/13.2"],
        extra_directives=["--mem=16G"],
        walltime="00:02:00",
    )
    script = _render(monkeypatch, tmp_path, config)
    directives = _directives(script)
    assert "#SBATCH --partition=roquo-debug" in directives
    assert "#SBATCH --account=rq-proj" in directives
    assert "#SBATCH --time=00:02:00" in directives
    # The reservation comes first among the additional directives.
    assert directives[-2:] == ["#SBATCH --reservation=rq-resv", "#SBATCH --mem=16G"]
    assert "module load cuda/13.2" in script.splitlines()


def test_gres_instead_of_gpus_per_node(monkeypatch, tmp_path):
    config = _example_config(extra_directives=["--gres=gpu:1"])
    del config["gpus_per_node"]
    directives = _directives(_render(monkeypatch, tmp_path, config))
    assert "#SBATCH --gres=gpu:1" in directives
    assert not any(d.startswith("#SBATCH --gpus-per-node") for d in directives)


def test_empty_account_and_single_launcher(monkeypatch, tmp_path):
    script = _render(monkeypatch, tmp_path, _example_config(account="", launcher="single"))
    assert not any(d.startswith("#SBATCH --account") for d in _directives(script))
    assert script.rstrip().endswith('"${QCSC_PREFECT_EXECUTABLE}" 0')
    assert "srun" not in script


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"account": "YOUR_ROQUO_ACCOUNT"}, "Set 'account'"),
        ({"partition": ""}, "Set 'partition'"),
        ({"gpus_per_node": 0}, "at least 1"),
        ({"queue": "roquo"}, "Unknown keys in the ROQUO config: queue"),
        ({"blocks": {"solver": "x"}}, r"Unknown keys in \[blocks\]: solver"),
    ],
)
def test_invalid_configs_fail_before_any_block_is_saved(changes, match):
    with pytest.raises(ValueError, match=match):
        create_blocks.build_blocks(_example_config(**changes))


def test_block_names_follow_the_config():
    blocks = create_blocks.build_blocks(_example_config(blocks={"hpc_profile": "hpc-roquo-mine"}))
    assert list(blocks) == ["cmd-roquo-gpu-hello", "exec-roquo-gpu-1", "hpc-roquo-mine"]
    hpc = blocks["hpc-roquo-mine"]
    assert hpc.hpc_target == "slurm"
    assert (hpc.queue_gpu, hpc.project_gpu) == ("roquo", "rq-test")


def _fake_nvidia_smi(bin_dir: Path, status: int) -> None:
    bin_dir.mkdir()
    tool = bin_dir / "nvidia-smi"
    tool.write_text(f"#!/bin/sh\necho '0, NVIDIA GB200, 189471 MiB'\nexit {status}\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)


def _run_hello(bin_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        ["sh", str(DEMO_DIR / "hello_gpu.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_hello_gpu_lists_the_gpu_and_exits_with_the_requested_status(tmp_path):
    _fake_nvidia_smi(tmp_path / "bin", 0)
    assert _run_hello(tmp_path / "bin").returncode == 0
    failed = _run_hello(tmp_path / "bin", "3")
    assert failed.returncode == 3
    assert "NVIDIA GB200" in failed.stdout


def test_hello_gpu_fails_when_no_gpu_can_be_listed(tmp_path):
    _fake_nvidia_smi(tmp_path / "bin", 9)
    result = _run_hello(tmp_path / "bin")
    assert result.returncode == 2
    assert "No GPU could be listed" in result.stderr


def test_flow_passes_the_exit_code_and_returns_the_job_record(monkeypatch, tmp_path):
    flow_module = _load_demo_module("flow")
    captured = {}

    class _Result:
        job_id = "4242"
        exit_status = 3
        state = "FAILED"
        job_status = {"NodeList": "rq001"}

    async def fake_run_job_from_blocks(**kwargs):
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(flow_module, "run_job_from_blocks", fake_run_job_from_blocks)
    record = asyncio.run(flow_module.roquo_gpu_hello_flow.fn(work_dir=str(tmp_path), exit_code=3))
    assert captured["user_args"] == ["3"]
    assert captured["hpc_profile_block_name"] == "hpc-roquo-gpu"
    assert record == {
        "job_id": "4242",
        "exit_status": 3,
        "state": "FAILED",
        "node_list": "rq001",
        "stdout_file": str(tmp_path.resolve() / "output.out"),
        "stderr_file": str(tmp_path.resolve() / "output.err"),
    }
