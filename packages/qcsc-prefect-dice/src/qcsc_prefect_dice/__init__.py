"""Shared DICE SHCI solver integration for qcsc-prefect workflows."""

from .block_utils import create_dice_blocks, register_dice_block_types
from .solver_job import DiceSHCISolverJob

__all__ = [
    "DiceSHCISolverJob",
    "create_dice_blocks",
    "register_dice_block_types",
]
