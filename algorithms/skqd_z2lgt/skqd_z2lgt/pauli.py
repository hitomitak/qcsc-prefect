"""Conversion of Pauli ops to JAX arrays."""
from typing import Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding
from jax.experimental.sparse import BCOO
from qiskit.quantum_info import SparsePauliOp

PAULI_ROW_INDICES = np.array([
    [0, 1],  # I
    [1, 0],  # X
    [1, 0],  # Y
    [0, 1]  # Z
], dtype=np.uint8)
PAULI_ELEMENTS = np.array([
    [1., 1.],  # I
    [1., 1.],  # X
    [1.j, -1.j],  # Y
    [1., -1.]  # Z
])


def to_bcoo(
    op: SparsePauliOp,
    indices: Optional[np.ndarray] = None
) -> BCOO:
    """Convert a SparsePauliOp to a sparse (COO) array, with optional subspace projection."""
    if indices is None:
        proj_dim = 2 ** op.num_qubits
    else:
        indices = np.asarray(indices)
        if len(indices.shape) == 1:
            # Convert integer indices to binary
            indices = (indices[:, None] >> np.arange(op.num_qubits)[None, ::-1]) % 2

        indices = jnp.array(indices)
        proj_dim = indices.shape[0]

    num_terms = len(op)
    if num_terms >= jax.device_count():
        num_dev = jax.device_count()
        terms_per_device = int(np.ceil(num_terms / num_dev).astype(int))
        pad_to_length = num_dev * terms_per_device
    else:
        num_dev = 1
        terms_per_device = num_terms
        pad_to_length = 0

    pauli_strings, op_coeffs = op_to_arrays(op, pad_to_length=pad_to_length)
    # Reshape for pmapping
    pauli_strings = pauli_strings.reshape((num_dev, terms_per_device, op.num_qubits))
    if num_dev > 1:
        mesh = jax.make_mesh((jax.device_count(), 1), ('device', 'dum'))
        pauli_strings = jax.device_put(pauli_strings, NamedSharding(mesh, PartitionSpec('device')))

    if indices is None:
        rows, mat_elems = rows_and_elements_all_vmap_pmap(pauli_strings)
    else:
        packed_indices = jnp.packbits(indices, axis=1)
        rows, mat_elems = rows_and_elements_scan_pmap(pauli_strings, indices, packed_indices)

    # Remove the device axis and truncate at the original number of op terms
    rows = rows.reshape((-1, proj_dim))[:num_terms].reshape(-1)
    cols = jnp.tile(jnp.arange(proj_dim), num_terms)
    data = (mat_elems.reshape((-1, proj_dim)) * op_coeffs[:, None])[:num_terms].reshape(-1)

    if indices is not None:
        filt = jnp.not_equal(rows, -1)
        data = data[filt]
        rows = rows[filt]
        cols = cols[filt]

    coords = jnp.stack([rows, cols], axis=1)
    return BCOO((data, coords), shape=(proj_dim, proj_dim))


def op_to_arrays(
    op: SparsePauliOp,
    pad_to_length: int = 0
) -> tuple[jax.Array, jax.Array]:
    """Convert Pauli strings into a 2D array of {0,1,2,3} indices."""
    pauli_index = {c: i for i, c in enumerate('IXYZ')}
    index_array = jnp.array([[pauli_index[c] for c in p.to_label()] for p in op.paulis],
                            dtype=np.uint8)
    coeff_array = jnp.array(op.coeffs)
    if (padding := pad_to_length - len(op)) > 0:
        index_array = jnp.concatenate(
            [index_array, jnp.zeros((padding, op.num_qubits), dtype=np.uint8)],
            axis=0
        )
        coeff_array = jnp.concatenate(
            [coeff_array, jnp.zeros(padding, dtype=np.complex128)],
            axis=0
        )

    return index_array, coeff_array


@jax.jit
def rows_and_elements_all(pauli_string: jax.Array) -> jax.Array:
    """Return the binary index of rows with nonzero elements in the Pauli matrix."""
    num_qubits = pauli_string.shape[0]
    # Can only work for num_qubits < 64 but practical limit for enumerating all rows is much lower
    rows = jnp.zeros((1,) * num_qubits, dtype=np.int64)
    elements = jnp.ones((1,) * num_qubits, dtype=np.complex128)
    pauli_row_indices = jnp.array(PAULI_ROW_INDICES, dtype=np.uint64)
    pauli_elements = jnp.array(PAULI_ELEMENTS)
    for iq in range(num_qubits):
        ex_dim = list(range(num_qubits))
        ex_dim.remove(iq)
        pidx = pauli_string[iq]
        rows += jnp.expand_dims(pauli_row_indices[pidx], ex_dim) * (1 << (num_qubits - 1 - iq))
        elements *= jnp.expand_dims(pauli_elements[pidx], ex_dim)

    return rows.reshape(-1), elements.reshape(-1)


rows_and_elements_all_vmap = jax.jit(jax.vmap(rows_and_elements_all))
rows_and_elements_all_vmap_pmap = jax.pmap(rows_and_elements_all_vmap, axis_name='device')


@jax.jit
def rows_and_elements(
    pauli_string: jax.Array,
    indices: jax.Array,
    packed_indices: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return the row numbers and matrix elements of the Pauli strings for the given columns."""
    mat_elems = matrix_element_vmap(pauli_string, indices)

    if pauli_string.shape[0] > 48:
        position_fn = bitstring_position_scan
    else:
        position_fn = bitstring_position_vmap

    # If pauli_string is all diagonal, mapped == indices
    rows = jax.lax.cond(
        jnp.all(jnp.equal(pauli_string % 3, 0)),
        lambda: jnp.arange(indices.shape[0]),
        lambda: position_fn(index_map_vmap(pauli_string, indices), packed_indices)
    )
    return rows, mat_elems


@jax.jit
def rows_and_elements_scan(
    pauli_strings: jax.Array,
    indices: jax.Array,
    packed_indices: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Temporally vectorized rows_and_elements. Cannot vmap because of memory footprint."""
    def body_fn(carry, pauli_string):
        _indices, _packed_indices = carry
        rows, mat_elems = rows_and_elements(pauli_string, _indices, _packed_indices)
        return carry, (rows, mat_elems)

    return jax.lax.scan(
        body_fn, (indices, packed_indices), pauli_strings
    )[1]


rows_and_elements_scan_pmap = jax.pmap(rows_and_elements_scan, axis_name='device',
                                       in_axes=(0, None, None))


@jax.jit
def bitstring_position(bitstring: jax.Array, pool: jax.Array) -> int:
    packed_bitstring = jnp.packbits(bitstring)
    matches = jnp.all(jnp.equal(packed_bitstring, pool), axis=1)
    idx = jnp.argmax(matches)
    return jax.lax.select(jnp.equal(idx, 0), jax.lax.select(matches[0], 0, -1), idx)


@jax.jit
def bitstring_position_scan(bitstrings: jax.Array, pool: jax.Array) -> jax.Array:
    def body_fn(_pool, bitstring):
        idx = bitstring_position(bitstring, _pool)
        return _pool, idx

    return jax.lax.scan(
        body_fn, pool, bitstrings
    )[1]


bitstring_position_vmap = jax.jit(jax.vmap(bitstring_position, in_axes=(0, None)))


@jax.jit
def pauli_signatures(pauli_strings: jax.Array) -> jax.Array:
    """Compute the Pauli signatures (diagonality) and return the result of np.unique."""
    signatures = jnp.sum(
        ((jnp.equal(pauli_strings, 0) | jnp.equal(pauli_strings, 3)).astype(int)
         * (1 << np.arange(pauli_strings.shape[-1])[None, ::-1])),
        axis=1
    )
    return jnp.unique(signatures, return_index=True, return_inverse=True)


@jax.jit
def reduce_rows_and_elements(
    uniq_return: tuple[jax.Array, jax.Array, jax.Array],
    coeffs: jax.Array,
    rows: jax.Array,
    elements: jax.Array
) -> tuple[jax.Array, jax.Array]:
    _, indices, inverse = uniq_return
    elements_reduced = jnp.zeros((indices.shape[0], elements.shape[-1]), dtype=elements.dtype)
    for idx, coeff, elems in zip(inverse, coeffs, elements):
        elements_reduced = elements_reduced.at[idx].add(coeff * elems)

    return rows[indices], elements_reduced


@jax.jit
def index_map(
    pauli_string: jax.Array,
    index: jax.Array
) -> jax.Array:
    """Return the row index of the nonzero entry in the Pauli matrix for the given column.

    Args:
        pauli_string: Pauli string represented as an array of {0,1,2,3} indices.
        index: A binary column index.

    Returns:
        The row index of the nonzero entry of pauli_string at column index.
    """
    # Mapped index
    return jnp.array(PAULI_ROW_INDICES)[pauli_string, index]


index_map_vmap = jax.jit(jax.vmap(index_map, in_axes=(None, 0)))


@jax.jit
def matrix_element(
    pauli_string: jax.Array,
    index: jax.Array
) -> jax.Array:
    """Return the matrix element of the nonzero entry in the Pauli matrix for the given column.

    Args:
        pauli_string: Pauli string represented as an array of {0,1,2,3} indices.
        index: A binary column index.

    Returns:
        The matrix element of the nonzero entry of pauli_string at column index.
    """
    # Matrix element product
    return jnp.prod(jnp.array(PAULI_ELEMENTS)[pauli_string, index])


matrix_element_vmap = jax.jit(jax.vmap(matrix_element, in_axes=(None, 0)))


@jax.jit
def apply_pauli(vector, rows, elements):
    return (vector * jnp.expand_dims(elements, tuple(range(1, vector.ndim))))[rows]


@jax.jit
def apply_paulis(vector, rows, elements):
    return jnp.sum(jax.vmap(apply_pauli, in_axes=(None, 0, 0))(vector, rows, elements), axis=0)


@jax.jit
def apply_i(vector):
    return vector


@jax.jit
def apply_x(vector):
    return vector[::-1]


@jax.jit
def apply_y(vector):
    vector = vector[::-1] * 1.j
    return vector.at[0].multiply(-1.)


@jax.jit
def apply_z(vector):
    return vector.at[1].multiply(-1.)


@partial(jax.jit, static_argnames=['qubit'])
def apply_pauli_qubit(vector, pauli_index, *, qubit):
    vector = jnp.moveaxis(vector, qubit, 0)
    vector = jax.lax.switch(pauli_index, [apply_i, apply_x, apply_y, apply_z], vector)
    return jnp.moveaxis(vector, 0, qubit)


@jax.jit
def apply_pauli_string(vector, pauli_string):
    nq = pauli_string.shape[0]
    fns = [partial(apply_pauli_qubit, qubit=i) for i in range(nq)]
    vector = vector.reshape((2,) * nq)
    vector = jax.lax.fori_loop(0, nq,
                               lambda i, x: jax.lax.switch(i, fns, x, pauli_string[i]),
                               vector)
    return vector.reshape(-1)


apply_paulistring_mat = jax.jit(jax.vmap(apply_pauli_string, in_axes=(1, None), out_axes=1))
apply_paulistrings = jax.jit(jax.vmap(apply_pauli_string, in_axes=(None, 0)))
apply_paulistrings_mat = jax.jit(jax.vmap(apply_paulistrings, in_axes=(1, None), out_axes=1))


@jax.jit
def apply_paulistring_sum(vector, pauli_strings, coeffs):
    def accumulate(iloop, vec):
        return vec + coeffs[iloop] * apply_pauli_string(vector, pauli_strings[iloop])

    return jax.lax.fori_loop(0, pauli_strings.shape[0], accumulate, jnp.zeros_like(vector))


apply_paulistring_sum_mat = jax.jit(jax.vmap(apply_paulistring_sum,
                                             in_axes=(1, None, None), out_axes=1))
