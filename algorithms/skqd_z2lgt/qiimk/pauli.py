"""Conversion of Pauli ops to JAX arrays."""
from typing import Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from qiskit.quantum_info import SparsePauliOp

PAULI_ROW_INDICES = np.array([
    [0, 1],  # I
    [1, 0],  # X
    [1, 0],  # Y
    [0, 1]  # Z
], dtype=np.int64)
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
    terms_per_device = int(np.ceil(num_terms / jax.device_count()).astype(int))
    pad_to_length = jax.device_count() * terms_per_device
    pauli_strings, op_coeffs = op_to_arrays(op, pad_to_length=pad_to_length)
    # Reshape for pmapping
    pauli_strings = pauli_strings.reshape((jax.device_count(), terms_per_device, op.num_qubits))
    if indices is None:
        rows, mat_elems = pv_rows_and_elements_all(pauli_strings)
    else:
        rows, mat_elems = pv_rows_and_elements(pauli_strings, indices)

    # Remove the device axis and truncate at the original number of op terms
    rows = rows.reshape((-1, proj_dim))[:num_terms]
    cols = jnp.tile(jnp.arange(proj_dim), num_terms)
    data = (mat_elems.reshape((-1, proj_dim)) * op_coeffs[:, None])[:num_terms].reshape(-1)

    if indices is None:
        cols = jnp.tile(jnp.arange(proj_dim), num_terms)
    else:
        rows = rows.reshape(-1)
        filt = jnp.not_equal(rows, -1)
        data = data[filt]
        rows = rows[filt]
        cols = cols[filt]

    indices = jnp.stack([rows, cols], axis=1)
    return BCOO((data, indices), shape=(proj_dim, proj_dim))


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
    pauli_row_indices = jnp.array(PAULI_ROW_INDICES)
    pauli_elements = jnp.array(PAULI_ELEMENTS)
    for iq in range(num_qubits):
        ex_dim = list(range(num_qubits))
        ex_dim.remove(iq)
        pidx = pauli_string[iq]
        rows += jnp.expand_dims(pauli_row_indices[pidx], ex_dim) * (1 << (num_qubits - 1 - iq))
        elements *= jnp.expand_dims(pauli_elements[pidx], ex_dim)

    return rows.reshape(-1), elements.reshape(-1)


v_rows_and_elements_all = jax.jit(jax.vmap(rows_and_elements_all))
pv_rows_and_elements_all = jax.pmap(v_rows_and_elements_all, axis_name='device')


@jax.jit
def rows_and_elements(
    pauli_string: jax.Array,
    indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the row numbers and matrix elements of the Pauli strings for the given columns."""
    mapped, mat_elems = v_pauli_map(pauli_string, indices)
    # mapped shape: [num_indices, num_qubits]
    rows = v_bitstring_position(mapped, indices)
    return rows, mat_elems


v_rows_and_elements = jax.jit(jax.vmap(rows_and_elements, in_axes=(0, None)))
pv_rows_and_elements = jax.pmap(v_rows_and_elements, axis_name='device', in_axes=(0, None))


@jax.jit
def bitstring_position(bitstring: jax.Array, pool: jax.Array) -> int:
    matches = jnp.all(jnp.equal(bitstring[None, :], pool), axis=1)
    idx = jnp.argmax(matches)
    return jax.lax.select(jnp.equal(idx, 0), jax.lax.select(matches[0], 0, -1), idx)


v_bitstring_position = jax.jit(jax.vmap(bitstring_position, in_axes=(0, None)))


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
def pauli_map(
    pauli_string: jax.Array,
    index: jax.Array
) -> tuple[jax.Array, complex]:
    """Return the row index and matrix element of the nonzero entry in the Pauli matrix for the
    given column.

    Args:
        index: A binary column index.
        pauli_string: Pauli string represented as an array of {0,1,2,3} indices.

    Returns:
        The row index and the matrix element of the nonzero entry of pauli_string at column index.
    """
    # Mapped index
    row_index = jnp.array(PAULI_ROW_INDICES)[pauli_string, index]
    # Matrix element product
    mat_elem = jnp.prod(jnp.array(PAULI_ELEMENTS)[pauli_string, index])
    return row_index, mat_elem


v_pauli_map = jax.jit(jax.vmap(pauli_map, in_axes=(None, 0)))


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
