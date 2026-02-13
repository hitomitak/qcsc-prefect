"""Fugaku-specific blocks (re-exports from common)."""

from hpc_prefect_blocks.common.blocks import (
    CommandBlock,
    ExecutionProfileBlock,
    HPCProfileBlock,
)

__all__ = [
    "CommandBlock",
    "ExecutionProfileBlock",
    "HPCProfileBlock",
]
