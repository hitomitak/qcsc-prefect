"""GB SQD Prefect workflow integration."""

__version__ = "0.1.0"

__all__ = [
    "ext_sqd_flow",
    "trim_sqd_flow",
    "ext_sqd_simple_flow",
    "trim_sqd_simple_flow",
    "bulk_gb_sqd_flow",
]


def __getattr__(name: str):
    if name in {"ext_sqd_flow", "trim_sqd_flow", "ext_sqd_simple_flow", "trim_sqd_simple_flow"}:
        from .main import (
            ext_sqd_flow,
            trim_sqd_flow,
            ext_sqd_simple_flow,
            trim_sqd_simple_flow,
        )

        return {
            "ext_sqd_flow": ext_sqd_flow,
            "trim_sqd_flow": trim_sqd_flow,
            "ext_sqd_simple_flow": ext_sqd_simple_flow,
            "trim_sqd_simple_flow": trim_sqd_simple_flow,
        }[name]
    if name == "bulk_gb_sqd_flow":
        from .bulk import bulk_gb_sqd_flow

        return bulk_gb_sqd_flow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
