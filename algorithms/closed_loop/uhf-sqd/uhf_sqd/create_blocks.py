"""Create the Prefect blocks of the UHF-SQD workflow from a TOML or JSON file.

``[solver]`` holds keyword arguments of ``qcsc_prefect_sbd.create_sbd_blocks``. ``integrals`` is
always ``"unrestricted"``, so the blocks get the UHF default names (``sbd-uhf-solver`` and so on)
unless names are given. Davidson settings left out take the defaults of the source's
``create_blocks.py`` (kunalroy332/qcsc-prefect at 3968e58), listed in ``SOLVER_DEFAULTS``. They
are loose and do not converge to the exact energy of the subspace; see the README.

``[quantum_sources.<name>]`` tables are saved as ``QuantumSourceBlock`` documents of that name,
for example ``[quantum_sources.uhf-sqd-random] config = { kind = "random" }``. An existing
Block of the same name (which another workflow may use) is replaced only with
``overwrite = true`` in its table; otherwise creation stops before saving it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

SOLVER_DEFAULTS = {
    "block": 4,
    "iteration": 1,
    "tolerance": 1e-2,
    "carryover_ratio": 0.1,
    "carryover_type": 0,
}


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return tomllib.loads(text)


def create_blocks(config: dict[str, Any]) -> dict[str, Any]:
    """Save the solver blocks and the quantum source blocks; return their names."""
    from qcsc_prefect.integrations.qiskit.sources import QuantumSourceBlock
    from qcsc_prefect_sbd import create_sbd_blocks

    unknown = set(config) - {"solver", "quantum_sources"}
    if unknown:
        raise ValueError(f"Unknown sections: {', '.join(sorted(unknown))}")
    solver = dict(config.get("solver", {}))
    if solver.get("integrals", "unrestricted") != "unrestricted":
        raise ValueError("UHF-SQD solver blocks read unrestricted integrals; remove 'integrals'.")
    solver["integrals"] = "unrestricted"
    names = create_sbd_blocks(**{**SOLVER_DEFAULTS, **solver})

    sources = {}
    for name, settings in config.get("quantum_sources", {}).items():
        settings = dict(settings)
        overwrite = settings.pop("overwrite", False)
        if not isinstance(overwrite, bool):
            raise ValueError(f"quantum_sources.{name}.overwrite must be true or false")
        block = QuantumSourceBlock.model_validate(settings)
        try:
            block.save(name, overwrite=overwrite)
        except ValueError as exc:
            if overwrite:
                raise
            raise ValueError(
                f"A QuantumSourceBlock named {name!r} already exists and may be used by another "
                f"workflow. Use another name, refer to it by name from the flow, or set "
                f"overwrite = true in [quantum_sources.{name}] to replace it."
            ) from exc
        sources[name] = block.config.kind
    return {**names, "quantum_sources": sources}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="TOML or JSON file")
    args = parser.parse_args(argv)
    json.dump(create_blocks(load_config(args.config)), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
