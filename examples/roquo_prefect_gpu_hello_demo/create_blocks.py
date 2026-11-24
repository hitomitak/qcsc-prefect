"""Create the Prefect Blocks for a one-GPU ROQUO job on the shared Slurm executor.

    uv run python examples/roquo_prefect_gpu_hello_demo/create_blocks.py \
        --config roquo_gpu.toml [--account ACCOUNT] [--partition NAME] [--reservation NAME]

ROQUO runs Slurm, so the HPC profile uses ``hpc_target="slurm"`` and the job goes through the
same executor as any Slurm cluster. Values on the command line override those in the file.
The Blocks are saved under the names in the file's ``[blocks]`` table, replacing Blocks of
the same names.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from prefect.blocks.core import Block
from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from qcsc_prefect_core.models.gpu_request import validate_gpu_request

DEMO_DIR = Path(__file__).resolve().parent
COMMAND_NAME = "roquo-gpu-hello"
EXECUTABLE_KEY = "roquo_gpu_hello"
PLACEHOLDER_ACCOUNT = "YOUR_ROQUO_ACCOUNT"
DEFAULT_BLOCK_NAMES = {
    "command": "cmd-roquo-gpu-hello",
    "execution_profile": "exec-roquo-gpu-1",
    "hpc_profile": "hpc-roquo-gpu",
}
CONFIG_KEYS = {
    "account",
    "partition",
    "reservation",
    "gpus_per_node",
    "walltime",
    "launcher",
    "modules",
    "extra_directives",
    "executable",
    "blocks",
}


def build_blocks(config: dict[str, Any]) -> dict[str, Block]:
    """The three Blocks for ``config``, keyed by the names they are saved under.

    Raises:
        ValueError: For an unknown key, the placeholder account, a missing partition, or a
            GPU request that ``hpc_target="slurm"`` cannot take.
    """
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in the ROQUO config: {', '.join(unknown)}.")
    unknown_names = sorted(set(config.get("blocks", {})) - set(DEFAULT_BLOCK_NAMES))
    if unknown_names:
        raise ValueError(f"Unknown keys in [blocks]: {', '.join(unknown_names)}.")
    names = {**DEFAULT_BLOCK_NAMES, **config.get("blocks", {})}

    account = str(config.get("account", "")).strip()
    if account == PLACEHOLDER_ACCOUNT:
        raise ValueError(
            f"Set 'account' to your ROQUO Slurm account (not {PLACEHOLDER_ACCOUNT!r}), "
            "or to '' for your default account."
        )
    partition = str(config.get("partition", "")).strip()
    if not partition:
        raise ValueError("Set 'partition' to the ROQUO GPU partition (see sinfo).")
    directives = []
    if reservation := str(config.get("reservation", "")).strip():
        directives.append(f"--reservation={reservation}")
    directives.extend(str(d) for d in config.get("extra_directives", []))

    gpus_per_node = config.get("gpus_per_node")
    validate_gpu_request(hpc_target="slurm", resource_class="gpu", gpus_per_node=gpus_per_node)
    executable = Path(config.get("executable") or DEMO_DIR / "hello_gpu.sh").expanduser()

    return {
        names["command"]: CommandBlock(
            command_name=COMMAND_NAME,
            executable_key=EXECUTABLE_KEY,
            description="One-GPU check for ROQUO",
            default_args=[],
        ),
        names["execution_profile"]: ExecutionProfileBlock(
            profile_name="roquo-gpu-1",
            command_name=COMMAND_NAME,
            resource_class="gpu",
            num_nodes=1,
            mpiprocs=1,
            gpus_per_node=gpus_per_node,
            walltime=str(config.get("walltime", "00:05:00")),
            launcher=config.get("launcher", "srun"),
            modules=[str(m) for m in config.get("modules", [])],
        ),
        names["hpc_profile"]: HPCProfileBlock(
            hpc_target="slurm",
            queue_cpu=partition,
            queue_gpu=partition,
            project_cpu=account,
            project_gpu=account,
            executable_map={EXECUTABLE_KEY: str(executable.resolve())},
            slurm_extra_directives=directives,
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="TOML file, see the example")
    parser.add_argument("--account", help="Slurm account; overrides the file")
    parser.add_argument("--partition", help="Slurm partition; overrides the file")
    parser.add_argument("--reservation", help="Slurm reservation; overrides the file")
    args = parser.parse_args(argv)

    with args.config.open("rb") as f:
        config = tomllib.load(f)
    for key in ("account", "partition", "reservation"):
        if (value := getattr(args, key)) is not None:
            config[key] = value

    blocks = build_blocks(config)
    for block in blocks.values():
        type(block).register_type_and_schema()
    for name, block in blocks.items():
        block.save(name, overwrite=True)
    print("Saved blocks: " + ", ".join(blocks))


if __name__ == "__main__":
    main()
