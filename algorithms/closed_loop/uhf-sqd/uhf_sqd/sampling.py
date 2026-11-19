"""Sampling policy of the UHF-SQD walker, independent of the platform a source Block selects."""

from __future__ import annotations

from qcsc_prefect.integrations.qiskit.sources import QuantumSource, QuantumSourceBlock

from .transpilation import TRANSPILER, prepare, transpile_config


def walker_seed(random_seed: int, trial_index: int, walker_index: int) -> int:
    """Seed of one walker: ``random_seed`` plus a pairing of the two indices.

    The source (3968e58) uses it for the random bitstrings and for the walker's recovery and
    subsampling generator, so that random and device sampling are controlled the same way.
    """
    return int(
        random_seed
        + ((trial_index + walker_index) * (trial_index + walker_index + 1) // 2 + walker_index)
    )


def load_quantum_source(
    quantum_source: str,
    *,
    circuit_params,
    num_orbitals: int,
    random_seed: int,
    trial_index: int,
    walker_index: int,
) -> QuantumSource:
    """Load the named source Block inside the walker and supply the UHF-SQD policy.

    ``quantum_source`` is an arbitrary Block document name. The Block owns source selection and
    platform settings; UHF-SQD supplies the width (2 * norb), the walker seed and the compile
    hooks. A random source draws the same bits as the source's
    ``generate_bit_array_uniform(shots, 2 * norb, seed)``.
    """
    return QuantumSourceBlock.load_source(
        quantum_source,
        num_bits=2 * num_orbitals,
        seed=walker_seed(random_seed, trial_index, walker_index),
        prepare=prepare,
        transpiler=TRANSPILER,
        transpile_config=transpile_config(circuit_params),
    )
