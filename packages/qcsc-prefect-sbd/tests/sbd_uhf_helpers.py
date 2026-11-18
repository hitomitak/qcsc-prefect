"""Shared test data for the unrestricted SBD tests."""

from __future__ import annotations

import numpy as np


def uhf_tensors(norb: int, seed: int = 7):
    """Random real integrals with the permutation symmetry of real orbitals."""
    rng = np.random.default_rng(seed)

    def h1():
        a = rng.normal(size=(norb, norb))
        return a + a.T

    def same_spin():
        t = rng.normal(size=(norb,) * 4)
        t = t + t.transpose(1, 0, 2, 3)
        t = t + t.transpose(0, 1, 3, 2)
        return t + t.transpose(2, 3, 0, 1)

    ab = rng.normal(size=(norb,) * 4)
    ab = ab + ab.transpose(1, 0, 2, 3)
    ab = ab + ab.transpose(0, 1, 3, 2)
    return {"h1_a": h1(), "h1_b": h1(), "h2_aa": same_spin(), "h2_ab": ab, "h2_bb": same_spin()}
