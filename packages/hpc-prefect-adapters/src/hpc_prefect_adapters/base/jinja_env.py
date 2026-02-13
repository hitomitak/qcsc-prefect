from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined

def make_env(pkg: str) -> Environment:
    """
    pkg: e.g. "hpc_prefect_adapters.miyabi"
    """
    env = Environment(
        loader=PackageLoader(pkg, "templates")
    )
    return env
