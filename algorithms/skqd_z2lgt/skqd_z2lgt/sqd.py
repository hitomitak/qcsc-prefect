"""Sample-based Krylov quantum diagonalization."""
from collections.abc import Iterable
from functools import partial
import logging
import time
from typing import Optional
import numpy as np
from scipy.sparse import csr_array, coo_array
from scipy.sparse.linalg import eigsh
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, bcoo_dot_general
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_sqd.qubit import sort_and_remove_duplicates, project_operator_to_subspace
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard
from skqd_z2lgt.pauli import op_to_arrays, multi_pauli_map

LOG = logging.getLogger(__name__)


def keys_to_intset(keys: Iterable[str]) -> set:
    keys_arr = np.array([list(map(int, key)) for key in keys])
    return set(np.sum(keys_arr * (1 << np.arange(keys_arr.shape[1])[::-1]), axis=1).tolist())


def sqd(
    hamiltonian: SparsePauliOp,
    states: np.ndarray,
    jax_device_id: Optional[int] = None
) -> tuple[np.ndarray, csr_array, float, np.ndarray]:
    """Perform a sample-based quantum diagonalization of the Hamiltonian.

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            [subspace_dim, num_qubits].
        jax_device_id: Index of the GPU device to use. If -1, the projection function is pmapped
            across all available GPUs.

    Returns:
        Array of sorted unique states, projected Hamiltonian as a CSR array, calculated ground state
        energy, and ground state vector.
    """
    pmap = False
    if jax_device_id is None:
        device = None
    elif jax_device_id < 0:
        device = None
        pmap = True
    else:
        device = jax.devices()[jax_device_id]

    with jax.default_device(device):
        start = time.time()
        states = jnp.packbits(states, axis=1)
        states = uniquify_states(states)
        LOG.info('%f seconds to sort %d bitstrings', time.time() - start, states.shape[0])
        start = time.time()
        hproj = to_bcoo(hamiltonian, states, pmap=pmap)
        LOG.info('%f seconds to project the Hamiltonian onto subspace', time.time() - start)
        start = time.time()
        eigval, eigvec = ground_state_lobpcg(hproj)
        LOG.info('%f seconds to diagonalize', time.time() - start)

    sqd_states = np.array(jnp.unpackbits(states, axis=1)[:, :hamiltonian.num_qubits])
    filt = jnp.logical_not(jnp.isclose(hproj.data, 0.))
    coo = coo_array(
        (
            hproj.data[filt],
            (hproj.indices[filt, 0], hproj.indices[filt, 1])
        ),
        shape=hproj.shape
    )
    ham_proj = csr_array(coo)

    return sqd_states, ham_proj, float(eigval), np.array(eigvec)


def uniquify_states(states: np.ndarray, num_qubits: Optional[int] = None) -> jax.Array:
    """Uniquify the states array. Note that np.unique performs the desired lexicographic sort."""
    states = jnp.unique(states, axis=0)
    if states.ndim == 1:
        # Convert integer indices to binary
        states = (states[:, None] >> jnp.arange(num_qubits)[None, ::-1]) % 2

    return states.astype(np.uint8)


def to_bcoo(
    op: SparsePauliOp,
    states: Optional[jax.Array] = None,
    pmap: bool = False
) -> BCOO:
    """Convert a SparsePauliOp to a sparse (COO) array, with optional subspace projection.

    Args:
        op: Sum of Pauli strings.
        states: Sorted list of unique packed bitstrings with shape [subspace_dim,
            ceil(num_qubits / 8)].
        pmap: Whether to map the projection function across GPU devices.

    Returns:
        A COO array encoding the op projected onto the subspace.
    """
    num_terms = len(op)
    if pmap:
        num_dev = jax.device_count()
        terms_per_device = int(np.ceil(num_terms / num_dev).astype(int))
        pad_to_length = num_dev * terms_per_device
        pauli_strings, op_coeffs = op_to_arrays(op, pad_to_length=pad_to_length)
        pauli_strings = pauli_strings.reshape((num_dev, 1, terms_per_device, op.num_qubits))
    else:
        pauli_strings, op_coeffs = op_to_arrays(op)

    # Possible extension: Adjust the pauli_strings shape when the next line fails with OOM
    rows, signs, imaginary = multi_pauli_map(pauli_strings, states)
    data, coords, shape = _make_bcoo_data(rows, signs, imaginary, op_coeffs, num_terms)

    if states is not None:
        filt = jnp.not_equal(coords[:, 0], -1)
        data = data[filt]
        coords = coords[filt]

    return BCOO((data, coords), shape=shape)


@partial(jax.jit, static_argnums=[4])
def _make_bcoo_data(rows, signs, imaginary, op_coeffs, num_terms):
    """Truncate at the original number of op terms and flatten."""
    subspace_dim = rows.shape[-1]
    phases = jnp.array([1., 1.j])[imaginary]
    data = (op_coeffs * phases)[:, None] * (1. - 2. * signs)
    if num_terms != rows.shape[0]:
        rows = rows[:num_terms]
        data = data[:num_terms]
    coords = jnp.empty(rows.shape + (2,), dtype=rows.dtype)
    coords = coords.at[..., 0].set(rows)
    coords = coords.at[..., 1].set(jnp.arange(subspace_dim, dtype=rows.dtype)[None, :])
    coords = coords.reshape((-1, 2))
    data = data.reshape(-1)
    return data, coords, (subspace_dim, subspace_dim)


@jax.jit
def ground_state_lobpcg(mat: BCOO) -> tuple[jax.Array, jax.Array]:
    """Find the 0th eigenvalue and eigenvector of a BCOO matrix."""
    xmat = jnp.ones((mat.shape[0], 1), dtype=np.complex128)
    # pylint: disable-next=unbalanced-tuple-unpacking
    eigvals, eigvecs, _ = lobpcg_standard(
        lambda x, m: bcoo_dot_general(m, x, dimension_numbers=(([1], [0]), ([], []))),
        xmat,
        args=(-mat,)
    )
    return -eigvals[0], eigvecs[:, 0]


def qiskit_sqd(
    hamiltonian: SparsePauliOp,
    states: np.ndarray,
    jax_device_id=None
) -> tuple[np.ndarray, csr_array, float, np.ndarray]:
    if jax_device_id is None:
        device = None
    else:
        device = jax.devices()[jax_device_id]

    bitstring_matrix = states[:, ::-1].astype(bool)
    with jax.default_device(device):
        start = time.time()
        bitstring_matrix = sort_and_remove_duplicates(bitstring_matrix)
        ham_proj = project_operator_to_subspace(bitstring_matrix, hamiltonian)
        LOG.info('%f seconds to project onto %d-dim subspace',
                 time.time() - start, bitstring_matrix.shape[0])

    start = time.time()
    evals, evecs = eigsh(ham_proj, k=1, which='SA')
    LOG.info('%f seconds to diagonalize', time.time() - start)
    return bitstring_matrix.astype(np.uint8), ham_proj, evals[0], evecs[:, 0]
