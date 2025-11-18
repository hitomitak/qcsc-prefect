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
from skqd_z2lgt.pauli import op_to_arrays, _subspace_pauli_map_nondiagonal
from skqd_z2lgt.utils import shard_array_1d, read_bits

LOG = logging.getLogger(__name__)


def keys_to_intset(keys: Iterable[str]) -> set:
    keys_arr = np.array([list(map(int, key)) for key in keys])
    return set(np.sum(keys_arr * (1 << np.arange(keys_arr.shape[1])[::-1]), axis=1).tolist())


def sqd(
    hamiltonian: SparsePauliOp,
    states: np.ndarray,
    states_size: Optional[int] = None,
    return_states: bool = True,
    return_hproj: bool = True,
    jax_device_id: int | list[int] = 0
) -> tuple[np.ndarray, csr_array, float, np.ndarray]:
    """Perform a sample-based quantum diagonalization of the Hamiltonian.

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            [subspace_dim, num_qubits].
        states_size: Fix the size of the states array used in computation to the specified value so
            that compilation is not triggered at each call with slightly different array sizes.
        return_states: Whether to return the sorted uniquified states.
        return_hproj: Whether to return the projected Hamiltonian.

    Returns:
        Calculated ground state energy, ground state vector, sorted uniquified states (if
        return_states=True), and the projected Hamiltonian as a CSR array (if return_hproj=True).
    """
    if isinstance(jax_device_id, int):
        jax_device_id = [jax_device_id]

    if states_size is not None:
        if (pad_length := states_size - states.shape[0]) < 0:
            raise ValueError('states_size smaller than the states array length')

        LOG.debug('Padding the states array..')
        # Extend axis 1 by 1 bit for the padding flag
        states = np.concatenate([np.zeros((states.shape[0], 1), dtype=np.uint8), states], axis=1)
        # Then extend axis 0 to states_size (fill with ones)
        states = np.concatenate([states, np.ones((pad_length, states.shape[1]), dtype=np.uint8)],
                                axis=0)
        LOG.debug('Done. Array shape %s', states.shape)

        start = time.time()
        diag, nondiag = get_hamiltonian_arrays(hamiltonian, jax_device_id)
        with jax.default_device(jax.devices()[jax_device_id[0]]):
            # Add an identity Pauli at the padding bit
            # concatenate respects sharding
            diag_paulis = jnp.concatenate(
                [jnp.zeros((diag[0].shape[0], 1), dtype=diag[0].dtype), diag[0]],
                axis=-1
            )
            nondiag_paulis = jnp.concatenate(
                [jnp.zeros((nondiag[0].shape[0], 1), dtype=nondiag[0].dtype), nondiag[0]],
                axis=-1
            )
            retval_jax = _sqd_fixed((diag_paulis, diag[1]), (nondiag_paulis, nondiag[1]), states,
                                    diag_paulis.sharding.num_devices, return_states, return_hproj)
        subspace_dim = int(retval_jax[0])
        LOG.info('%f seconds for SQD. Subspace dimension %d', time.time() - start, subspace_dim)
        retval = (float(retval_jax[1]), np.array(retval_jax[2][:subspace_dim]))
        if return_states:
            states = np.unpackbits(retval_jax[3][:subspace_dim], axis=1)
            states = states[:, 1:hamiltonian.num_qubits + 1]
            retval += (states,)
        if return_hproj:
            ham = retval_jax[-1]
            retval += (bcoo_to_csr(BCOO((ham.data, ham.indices), shape=(subspace_dim,) * 2)),)
    else:
        start = time.time()
        with jax.default_device(jax.devices()[jax_device_id[0]]):
            states = jnp.packbits(states, axis=1)
            states = uniquify_states(states)
            LOG.info('%f seconds to sort %d bitstrings', time.time() - start, states.shape[0])
            start = time.time()
            hproj = to_bcoo(hamiltonian, states, jax_device_id)
            LOG.info('%f seconds to project the Hamiltonian onto subspace', time.time() - start)
            start = time.time()
            eigval, eigvec = ground_state_lobpcg(hproj)
        LOG.info('%f seconds to diagonalize', time.time() - start)

        retval = (float(eigval), np.array(eigvec))
        if return_states:
            states = read_bits(states, num_bits=hamiltonian.num_qubits)
            retval += (states,)
        if return_hproj:
            retval += (bcoo_to_csr(hproj),)

    return retval


@partial(jax.jit, static_argnums=[3, 4, 5])
def _sqd_fixed(
    diag,
    nondiag,
    states,
    num_devices,
    return_states,
    return_hproj
):
    states = jnp.packbits(states, axis=1)
    states = jnp.unique(states, axis=0, size=states.shape[0], fill_value=255)
    subspace_dim = jnp.searchsorted(states[:, 0] >> 7, 1)
    data, coords = _make_bcoo_data(diag, nondiag, states, num_devices)
    # rows[subspace_dim:] are either -1 or range(subspace_dim, rows.shape[0])
    mask = jnp.logical_and(jnp.not_equal(coords[:, 0], -1), jnp.less(coords[:, 0], subspace_dim))
    data *= mask
    coords *= mask[:, None]
    hproj = BCOO((data, coords), shape=(states.shape[0],) * 2)
    retval = ground_state_lobpcg(hproj)
    if return_states:
        retval += (states,)
    if return_hproj:
        retval += (hproj,)
    return (subspace_dim,) + retval


def uniquify_states(
    states: np.ndarray,
    num_qubits: Optional[int] = None,
    size: Optional[int] = None
) -> jax.Array:
    """Uniquify the states array. Note that np.unique performs the desired lexicographic sort."""
    states = jnp.unique(states, axis=0, size=size, fill_value=255)
    if states.ndim == 1:
        # Convert integer indices to binary
        states = (states[:, None] >> jnp.arange(num_qubits)[None, ::-1]) % 2

    return states.astype(np.uint8)


def get_hamiltonian_arrays(
    hamiltonian: SparsePauliOp,
    jax_device_id: int | list[int] = 0
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    if isinstance(jax_device_id, int):
        jax_device_id = [jax_device_id]

    with jax.default_device(jax.devices()[jax_device_id[0]]):
        pauli_strings, op_coeffs = op_to_arrays(hamiltonian)
        is_diag = jnp.all(jnp.equal(pauli_strings % 3, 0), axis=1)
        diag_paulis = pauli_strings[is_diag]
        diag_coeffs = op_coeffs[is_diag]
        nondiag_paulis = pauli_strings[jnp.logical_not(is_diag)]
        nondiag_coeffs = op_coeffs[jnp.logical_not(is_diag)]
    if len(jax_device_id) > 1:
        diag_paulis = shard_array_1d(diag_paulis, device_ids=jax_device_id)
        diag_coeffs = shard_array_1d(diag_coeffs, device_ids=jax_device_id)
        nondiag_paulis = shard_array_1d(nondiag_paulis, device_ids=jax_device_id)
        nondiag_coeffs = shard_array_1d(nondiag_coeffs, device_ids=jax_device_id)

    return (diag_paulis, diag_coeffs), (nondiag_paulis, nondiag_coeffs)


def to_bcoo(
    hamiltonian: SparsePauliOp,
    states: jax.Array,
    jax_device_id: int | list[int] = 0
) -> BCOO:
    """Convert a SparsePauliOp to a sparse (COO) array, with optional subspace projection.

    Args:
        op: Sum of Pauli strings.
        states: Sorted list of unique packed bitstrings with shape [subspace_dim,
            ceil(num_qubits / 8)].

    Returns:
        A COO array encoding the op projected onto the subspace.
    """
    if isinstance(jax_device_id, int):
        jax_device_id = [jax_device_id]
    diag, nondiag = get_hamiltonian_arrays(hamiltonian, jax_device_id)
    # Possible extension: Adjust the pauli_strings shape when the next line fails with OOM
    data, coords = _make_bcoo_data(diag, nondiag, states, diag[0].sharding.num_devices)

    mask = jnp.not_equal(coords[:, 0], -1)
    if jax.device_count() > 1:
        # Keep the sharding
        data *= mask
        coords *= mask[:, None]
    else:
        data = data[mask]
        coords = coords[mask]

    return BCOO((data, coords), shape=(states.shape[0],) * 2)


def bcoo_to_csr(bcoo: BCOO):
    filt = jnp.logical_not(jnp.isclose(bcoo.data, 0.))
    coo = coo_array(
        (
            bcoo.data[filt],
            (bcoo.indices[filt, 0], bcoo.indices[filt, 1])
        ),
        shape=bcoo.shape
    )
    return csr_array(coo)


_v_subspace_pauli_map_nondiagonal = jax.jit(
    jax.vmap(_subspace_pauli_map_nondiagonal, in_axes=(0, None))
)


@partial(jax.jit, static_argnums=[3])
def _make_bcoo_data(diag, nondiag, states, num_devices):
    """Return data, coords, and shape from the given representation of Paulis and states."""
    # Nondiagonal part
    nondiag_rows, signs = _v_subspace_pauli_map_nondiagonal(nondiag[0], states)
    imaginary = (jnp.sum(jnp.equal(nondiag[0], 2), axis=-1) % 2).astype(np.uint8)
    phases = jnp.array([1., 1.j])[imaginary]
    nondiag_data = (nondiag[1] * phases)[:, None] * (1. - 2. * signs)
    # Diagonal part
    is_signed = jnp.packbits(jnp.greater(diag[0], 1).astype(np.uint8), axis=1)
    signs = jnp.sum(jnp.bitwise_count(states[None, ...] & is_signed[:, None]), axis=-1) % 2
    diag_data = diag[1][:, None] * (1. - 2. * signs)

    aidx = jnp.arange(states.shape[0], dtype=np.int32)[None, :]

    if num_devices == 1:
        # Single-device
        data = jnp.concatenate([jnp.sum(diag_data, axis=0, keepdims=True), nondiag_data], axis=0)
        rows = jnp.concatenate([aidx, nondiag_rows], axis=0)
        cols = jnp.tile(aidx, (rows.shape[0], 1))
        coords = jnp.stack([rows, cols], axis=-1)
    else:
        # Sharded
        diag_data = diag_data.reshape((num_devices, -1, states.shape[0]))
        diag_data = jnp.sum(diag_data, axis=1, keepdims=True)
        nondiag_data = nondiag_data.reshape((num_devices, -1, states.shape[0]))
        data = jnp.concatenate([diag_data, nondiag_data], axis=1)
        nondiag_rows = nondiag_rows.reshape((num_devices, -1, states.shape[0]))
        diag_rows = jnp.tile(aidx[None, ...], (num_devices, 1, 1))
        rows = jnp.concatenate([diag_rows, nondiag_rows], axis=1)
        cols = jnp.tile(aidx[None, ...], rows.shape[:2] + (1,))
        coords = jnp.stack([rows, cols], axis=-1)

    data = data.reshape(-1)
    coords = coords.reshape((-1, 2))
    return data, coords


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
    states: np.ndarray
) -> tuple[np.ndarray, csr_array, float, np.ndarray]:
    bitstring_matrix = states[:, ::-1].astype(bool)
    start = time.time()
    bitstring_matrix = sort_and_remove_duplicates(bitstring_matrix)
    ham_proj = project_operator_to_subspace(bitstring_matrix, hamiltonian)
    LOG.info('%f seconds to project onto %d-dim subspace',
             time.time() - start, bitstring_matrix.shape[0])

    start = time.time()
    evals, evecs = eigsh(ham_proj, k=1, which='SA')
    LOG.info('%f seconds to diagonalize', time.time() - start)
    return evals[0], evecs[:, 0], bitstring_matrix.astype(np.uint8), ham_proj
