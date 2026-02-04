"""
Run-time parameters (Deployment UI).
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class Tuning(BaseModel):
    nodes: int | None = Field(default=None, gt=0, description="Override node count")
    walltime: str | None = Field(default=None, description="Override walltime HH:MM:SS")
    ranks_per_node: int | None = Field(default=None, gt=0, description="Override ranks per node")
    threads_per_rank: int | None = Field(default=None, gt=0, description="Override threads per rank")
