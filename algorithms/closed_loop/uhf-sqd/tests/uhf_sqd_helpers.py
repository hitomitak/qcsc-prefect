"""Helpers of the UHF-SQD tests: determinant encodings, a fake per-spin solver, a walker runner.

The walker runs without a Prefect server: its tasks are called directly and the logger and the
array storage are replaced.
"""

from __future__ import annotations

import dataclasses
import itertools
from types import SimpleNamespace

import numpy as np
from qcsc_prefect.integrations.qiskit.sources.random import RandomSource
from qcsc_prefect_sbd import SBDResult
from uhf_sqd import sqd
from uhf_sqd.flow_params import CircuitParameters, SamplerParameters
from uhf_sqd.sampling import walker_seed

NORB, NA, NB = 6, 3, 2
KEY = "v2|" + "ab" * 32 + "|af=None|nelec=(3, 2)"


def ints_to_rows(ints, norb):
    """CI-string ints -> (n, norb) bool rows (bit 0 = most significant), the carryover layout."""
    return np.array(
        [[(int(x) >> (norb - 1 - i)) & 1 for i in range(norb)] for x in ints], dtype=bool
    ).reshape(-1, norb)


def rows_to_ints(rows):
    rows = np.asarray(rows, dtype=bool)
    return [int("".join("1" if b else "0" for b in row), 2) for row in rows] if rows.size else []


def all_strings(norb, nelec):
    return sorted(sum(1 << p for p in c) for c in itertools.combinations(range(norb), nelec))


def random_strings(rng, norb, nelec, n):
    out = set()
    while len(out) < n:
        out.add(sum(1 << int(p) for p in rng.choice(norb, size=nelec, replace=False)))
    return sorted(out)


@dataclasses.dataclass
class Call:
    alpha: list
    beta: list
    kwargs: dict


class FakeSolver:
    """Stands in for SBDSolverJob.run. Like the real binary, the carryover it returns is a subset
    of the determinants it was given, with different lengths per spin, and it records every call."""

    root_dir = None
    integrals = "unrestricted"
    solver_mode = "cpu"

    def __init__(self, n_alpha_co=5, n_beta_co=2):
        self.calls: list[Call] = []
        self.n_alpha_co, self.n_beta_co = n_alpha_co, n_beta_co

    async def run(self, *, ci_strings, nelec, norb, **kwargs):
        a, b = np.asarray(ci_strings[0]), np.asarray(ci_strings[1])
        self.calls.append(Call(sorted(a.tolist()), sorted(b.tolist()), kwargs))
        na, nb = nelec
        return SBDResult(
            energy=-1.0 - 0.01 * len(self.calls),
            orbital_occupancies=(np.full(norb, na / norb), np.full(norb, nb / norb)),
            carryover_bitstrings=ints_to_rows(sorted(a.tolist())[: self.n_alpha_co], norb),
            carryover_bitstrings_b=ints_to_rows(sorted(b.tolist())[: self.n_beta_co], norb),
        )


def fake_props(norb=NORB, nelec=(NA, NB)):
    h1 = np.diag(np.arange(norb, dtype=float))
    h2 = np.zeros((norb,) * 4)
    na, nb = nelec
    return SimpleNamespace(
        num_orbitals=norb,
        num_electrons=nelec,
        unrestricted=True,
        open_shell=na != nb,
        one_body_tensor=h1,
        one_body_tensor_b=h1,
        two_body_tensor=h2,
        two_body_tensor_ab=h2,
        two_body_tensor_bb=h2,
        nuclear_repulsion_energy=0.0,
        spin_sq=None,
        initial_occupancy=(np.full(norb, na / norb), np.full(norb, nb / norb)),
        reference_tag=None,
        reference_source=None,
        fcidump_sha256=None,
        reference_cache_key=KEY,
    )


class Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    debug = exception = info

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message % args)


def prepare_walker(monkeypatch, *, solver, source=None, save_ndarray=None):
    """Replace the walker's Prefect calls, solver, source and (optionally) storage."""
    logger = Logger()
    monkeypatch.setattr(sqd, "get_run_logger", lambda: logger)
    monkeypatch.setattr(sqd.SBDSolverJob, "load", classmethod(lambda cls, name: solver))
    for name in ("recover_configurations", "postselect_bitstrings", "subsample_open_shell"):
        monkeypatch.setattr(sqd, name, getattr(sqd, name).fn)

    def load_source(quantum_source, *, num_orbitals, random_seed, trial_index, walker_index, **kw):
        if source is not None:
            return source
        return RandomSource(
            num_bits=2 * num_orbitals, seed=walker_seed(random_seed, trial_index, walker_index)
        )

    monkeypatch.setattr(sqd, "load_quantum_source", load_source)
    if save_ndarray is not None:
        monkeypatch.setattr(sqd, "save_ndarray", save_ndarray)
    return logger


def run_walker(
    *,
    n_steps=1,
    carryover=None,
    carryover_b=None,
    ep=None,
    sqd_dim=40,
    shots=400,
    n_batches=1,
    use_reset_mitigation=True,
):
    return sqd.walker_sqd.fn(
        trial_index=0,
        walker_index=0,
        ucj_parameter=np.zeros(3),
        circuit_params=CircuitParameters(use_reset_mitigation=use_reset_mitigation),
        elec_props=ep or fake_props(),
        aa_indices=[],
        ab_indices=[],
        carryover=np.full((0, NORB), False) if carryover is None else carryover,
        carryover_b=carryover_b,
        sqd_dim=sqd_dim,
        solver_block_name="x",
        quantum_source="test-source",
        random_seed=1,
        n_recovery_steps=n_steps,
        n_batches=n_batches,
        sampler_params=SamplerParameters(shots=shots),
    )
