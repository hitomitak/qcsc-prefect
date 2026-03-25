"""Sample-based Krylov quantum diagonalization."""
import logging
import time
from typing import Optional
import numpy as np
from scipy.sparse import csr_array, coo_array
import jax
import jax.numpy as jnp
from jax._src.numpy import array_creation
from jax._src.lax import lax
from qiskit.quantum_info import SparsePauliOp
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard
from skqd_z2lgt.utils import shard_array_1d, read_bits

LOG = logging.getLogger(__name__)


def sqd(
    hamiltonian: SparsePauliOp,
    states: np.ndarray,
    states_size: Optional[int] = None,
    return_states: bool = True,
    return_hproj: bool = True,
    jax_device_id: int | list[int] = 0
) -> tuple[float, np.ndarray, np.ndarray, csr_array]:
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

    with jax.default_device(jax.devices()[jax_device_id[0]]):
        paulis_d, coeffs_d, paulis_n, coeffs_n = get_hamiltonian_arrays(hamiltonian, jax_device_id,
                                                                        npmod=jnp)

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
        with jax.default_device(jax.devices()[jax_device_id[0]]):
            # Add an identity Pauli at the padding bit
            # concatenate respects sharding
            paulis_d = jnp.concatenate(
                [jnp.zeros((paulis_d.shape[0], 1), dtype=paulis_d.dtype), paulis_d],
                axis=-1
            )
            paulis_n = jnp.concatenate(
                [jnp.zeros((paulis_n.shape[0], 1), dtype=paulis_n.dtype), paulis_n],
                axis=-1
            )
            retval_jax = _sqd_fixed(paulis_d, coeffs_d, paulis_n, coeffs_n, states)
        subspace_dim = int(retval_jax[0])
        LOG.info('%f seconds for SQD. Subspace dimension %d', time.time() - start, subspace_dim)
        retval = (float(retval_jax[-2]), np.array(retval_jax[-1][:subspace_dim]))
        if return_states:
            sqd_states = read_bits(retval_jax[1][:subspace_dim], num_bits=hamiltonian.num_qubits,
                                   offset=1)
        if return_hproj:
            diagonals, rows, nondiag_data = map(np.asarray, retval_jax[2:5])
            diagonals = diagonals[:subspace_dim]
            num_n = paulis_n.shape[0]
            rows = rows.reshape(num_n, states_size)[:, :subspace_dim]
            data = nondiag_data.reshape(num_n, states_size)[:, :subspace_dim]
            nondiagonals = reduce_nondiagonals(rows, data)

    else:
        with jax.default_device(jax.devices()[jax_device_id[0]]):
            start = time.time()
            states_u = uniquify_states(states)
            LOG.info('%f seconds to sort %d bitstrings', time.time() - start, states_u.shape[0])
            start = time.time()
            diagonals = get_diagonals(paulis_d, coeffs_d, states_u)
            rows, data = get_nondiagonals(paulis_n, coeffs_n, states_u)
            nondiagonals = reduce_nondiagonals(rows, data, npmod=jnp)
            eigval, eigvec = compute_ground_state(diagonals, nondiagonals)
            LOG.info('%f seconds to diagonalize', time.time() - start)

        retval = (float(eigval), np.array(eigvec))
        if return_states:
            sqd_states = read_bits(states_u, num_bits=hamiltonian.num_qubits)
    if return_states:
        retval += (sqd_states,)  # pylint: disable=used-before-assignment
    if return_hproj:
        retval += (to_csr(diagonals, nondiagonals),)

    return retval


@jax.jit
def _sqd_fixed(paulis_d, coeffs_d, paulis_n, coeffs_n, states):
    states = uniquify_states(states, size=states.shape[0])
    subspace_dim = jnp.searchsorted(states[:, 0] >> 7, 1)
    diagonals = get_diagonals(paulis_d, coeffs_d, states)
    rows, nondiag_data = get_nondiagonals(paulis_n, coeffs_n, states)
    mask = jnp.logical_and(jnp.not_equal(rows, -1), jnp.less(rows, subspace_dim))
    rows *= mask
    nondiag_data *= mask
    cols = jnp.tile(jnp.arange(states.shape[0]), rows.shape[0])
    rows = rows.reshape(-1)
    nondiag_data = nondiag_data.reshape(-1)
    eigval, eigvec = compute_ground_state(diagonals, (cols, rows, nondiag_data))
    return (subspace_dim, states, diagonals, rows, nondiag_data, eigval, eigvec)


def uniquify_states(
    states: np.ndarray,
    size: Optional[int] = None
) -> jax.Array:
    """Uniquify the states array. Note that np.unique performs the desired lexicographic sort."""
    states = jnp.packbits(states, axis=1)
    states = jnp.unique(states, axis=0, size=size, fill_value=255)
    return states.astype(np.uint8)


def get_hamiltonian_arrays(
    hamiltonian: SparsePauliOp,
    device_ids: Optional[list[int]] = None,
    npmod=np
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    paulis = hamiltonian.paulis
    pauli_index = npmod.array([[0, 3], [1, 2]], dtype=np.uint8)
    pauli_strings = pauli_index[paulis.x[..., ::-1].astype(np.int32),
                                paulis.z[..., ::-1].astype(np.int32)]
    coeffs = npmod.array(hamiltonian.coeffs)
    is_nondiag = npmod.any(paulis.x, axis=1)
    nondiag_paulis = pauli_strings[is_nondiag]
    nondiag_coeffs = coeffs[is_nondiag]
    diag_paulis = pauli_strings[~is_nondiag]
    diag_coeffs = coeffs[~is_nondiag]

    if device_ids and len(device_ids) > 1:
        diag_paulis = shard_array_1d(diag_paulis, device_ids=device_ids)
        diag_coeffs = shard_array_1d(diag_coeffs, device_ids=device_ids)
        nondiag_paulis = shard_array_1d(nondiag_paulis, device_ids=device_ids)
        nondiag_coeffs = shard_array_1d(nondiag_coeffs, device_ids=device_ids)

    return diag_paulis, diag_coeffs, nondiag_paulis, nondiag_coeffs


@jax.jit
def get_diagonals(
    pauli_strings: jax.Array,
    coeffs: jax.Array,
    states: jax.Array
) -> jax.Array:
    """Compute the diagonals of the Hamiltonian.

    Args:
        pauli_strings: List of diagonal Pauli strings.
        coeffs: Coefficients of the Pauli strings.
        states: Bit-packed SQD subspace.

    Returns:
        An array of floats with size len(states).
    """
    is_signed = jnp.packbits((pauli_strings == 3).astype(np.uint8), axis=1)
    signs = jnp.sum(jnp.bitwise_count(states[None, ...] & is_signed[:, None]), axis=-1) % 2
    return jnp.sum(coeffs[:, None] * (1. - 2. * signs), axis=0).real


@jax.jit
def get_nondiagonals(
    pauli_strings: jax.Array,
    coeffs: jax.Array,
    states: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute the nondiagonal parts of the Hamiltonian.

    Args:
        pauli_strings: List of nondiagonal Pauli strings.
        coeffs: Coefficients of the Pauli strings.
        states: Bit-packed SQD subspace.

    Returns:
        Arrays of row indices and matrix elements of the Hamiltonian matrix projected onto the
        subspace (each shape [num_paulis, len(states)]). Nonexistent rows are indicated by index -1.
    """
    rows, signs = jax.vmap(subspace_pauli_map, in_axes=(0, None))(pauli_strings, states)
    phases = jnp.array([1., 1.j, -1., -1.j])[jnp.sum(pauli_strings == 2, axis=-1) % 4]
    data = (coeffs * phases)[:, None] * (1. - 2. * signs)
    return rows, data


@jax.jit
def subspace_pauli_map(pauli_string: jax.Array, states: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the row indices and signs of the matrix elements at given columns.

    Args:
        pauli_string: Nondiagonal Pauli string.
        states: Bit-packed SQD subspace.

    Returns:
        Arrays of row indices and signs of the matrix elements of the Pauli string projected onto
        the subspace. Nonexistent rows are indicated by index -1.
    """
    is_nondiagonal = jnp.packbits(jnp.not_equal(pauli_string % 3, 0).astype(np.uint8))
    is_signed = jnp.packbits(jnp.greater(pauli_string, 1).astype(np.uint8))
    mapped_states = states ^ is_nondiagonal
    rows = subspace_indices(mapped_states, states)
    signs = jnp.sum(jnp.bitwise_count(states & is_signed), axis=-1) % 2
    return rows, signs


@jax.jit
def subspace_indices(mapped_states: jax.Array, states: jax.Array) -> jax.Array:
    """Return the positions of the states in the subspace, or -1 if not found."""
    # Borrowing from jax._src.numpy.lax_numpy._searchsorted_via_sort
    # Attempted to replace the lax functions with corresponding jnp ones but that worsened the
    # memory leak by 2x
    def _rank(x):
        idx = lax.iota(np.int32, x.shape[0])
        # lax.sort seems to leak GPU memory; can lose as much as 5 GB when sorting x of shape (5M,9)
        sorted_idx = lax.sort(tuple(x.T) + (idx,), num_keys=x.shape[1])[-1]
        return array_creation.zeros_like(idx).at[sorted_idx].set(idx)

    index = _rank(lax.concatenate([mapped_states, states], 0))[:mapped_states.shape[0]]
    positions = lax.sub(index, _rank(mapped_states)).astype(np.int32)
    return jnp.where(jnp.all(jnp.equal(mapped_states, states[positions]), axis=1), positions, -1)


def reduce_nondiagonals(
    rows: np.ndarray,
    data: np.ndarray,
    npmod=np
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat arrays of columns, rows, and matrix elements of the nondiagonal Hamiltonian."""
    indices = npmod.nonzero(npmod.not_equal(rows, -1))
    return indices[1], rows[indices], data[indices]


@jax.jit
def apply_h(xmat, diagonals, idx_in, idx_out, data):
    result = xmat * diagonals[:, None]
    result = result.at[idx_out].add(xmat.at[idx_in].get() * data[:, None])
    return result


@jax.jit
def compute_ground_state(
    diagonals: jax.Array,
    nondiagonals: tuple[jax.Array, jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Find the 0th eigenvalue and eigenvector of a BCOO matrix."""
    idx_in, idx_out, data = nondiagonals
    xmat = jax.nn.one_hot(jnp.argmin(diagonals), diagonals.shape[0])[:, None].astype(np.complex128)
    # pylint: disable-next=unbalanced-tuple-unpacking
    eigvals, eigvecs, _ = lobpcg_standard(apply_h, xmat, args=(-diagonals, idx_in, idx_out, -data))
    return -eigvals[0], eigvecs[:, 0]


def to_csr(
    diagonals: np.ndarray,
    nondiagonals: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> csr_array:
    cols, rows, data = nondiagonals
    filt = np.logical_not(np.isclose(data, 0.))
    cols = cols[filt]
    rows = rows[filt]
    data = data[filt]
    cols = np.concatenate([cols, np.arange(diagonals.shape[0])])
    rows = np.concatenate([rows, np.arange(diagonals.shape[0])])
    data = np.concatenate([data, diagonals.astype(data.dtype)])
    coo = coo_array((data, (rows, cols)), shape=diagonals.shape * 2)
    return csr_array(coo)


def make_hproj(hamiltonian: SparsePauliOp, states: np.ndarray) -> csr_array:
    """Just compose the projected sparse Hamiltonian."""
    paulis_d, coeffs_d, paulis_n, coeffs_n = get_hamiltonian_arrays(hamiltonian)
    diagonals = get_diagonals(paulis_d, coeffs_d, states)
    rows, data = get_nondiagonals(paulis_n, coeffs_n, states)
    nondiagonals = reduce_nondiagonals(rows, data)
    return to_csr(diagonals, nondiagonals)
