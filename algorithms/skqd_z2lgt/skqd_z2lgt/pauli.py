# pylint: disable=cell-var-from-loop
"""Conversion of Pauli ops to JAX arrays."""
from typing import Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from qiskit.quantum_info import SparsePauliOp


def op_to_arrays(
    op: SparsePauliOp,
    pad_to_length: int = 0
) -> tuple[jax.Array, jax.Array]:
    """Convert Pauli strings into an array of {0,1,2,3} indices.

    Args:
        op: Sum of Pauli strings.
        pad_to_length: Zero-pad the returned arrays to specified length.

    Returns:
        Arrays of Pauli indices (shape [num_terms, num_qubits]) and coefficients ([num_terms]).
    """
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


def multi_pauli_map(
    pauli_strings: jax.Array,
    states: Optional[jax.Array] = None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if states is None:
        match pauli_strings.ndim:
            case 2:
                rows, signs = _v_pauli_map(pauli_strings)
            case 3:
                rows, signs = _pv_pauli_map(pauli_strings)
            case _:
                raise ValueError('Too many dimensions in pauli_strings')

        subspace_dim = 2 ** pauli_strings.shape[-1]
    else:
        subspace = jnp.packbits(states, axis=1)
        match pauli_strings.ndim:
            case 2:
                rows, signs = _v_subspace_pauli_map(pauli_strings, states, subspace)
            case 3:
                rows, signs = _sv_subspace_pauli_map(pauli_strings, states, subspace)
            case 4:
                rows, signs = _psv_subspace_pauli_map(pauli_strings, states, subspace)
            case _:
                raise ValueError('Too many dimensions in pauli_strings')

        subspace_dim = states.shape[0]

    imaginary = (jnp.sum(jnp.equal(pauli_strings, 2), axis=-1) % 2).astype(np.uint8)

    shape = (-1, subspace_dim)
    # pylint: disable-next=used-before-assignment
    return rows.reshape(shape), signs.reshape(shape), imaginary.reshape(-1)


@jax.jit
def _pauli_map(pauli_string: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the row indices and matrix element phases of nonzero entries of the Pauli unitary.

    Phases are integers in units of pi/2.

    This method only works for num_qubits < 64, but the practical limit for enumerating all rows is
    much lower.

    Args:
        pauli_string: Array of {0,1,2,3} pauli indices, shape [num_qubits].

    Returns:
        Row indices (shape [2 ** num_qubits]) and matrix element signs ([2 ** num_qubits]) of the
        nonzero entries of the Pauli unitary, ordered by columns.
    """
    return jax.lax.cond(
        jnp.all(jnp.equal(pauli_string % 3, 0)),
        _pauli_map_diagonal,
        _pauli_map_nondiagonal,
        pauli_string
    )


_v_pauli_map = jax.jit(jax.vmap(_pauli_map))
_pv_pauli_map = jax.pmap(_v_pauli_map, axis_name='device')


@jax.jit
def _pauli_map_diagonal(pauli_string: jax.Array) -> tuple[jax.Array, jax.Array]:
    num_qubits = pauli_string.shape[0]
    rows = jnp.arange(2 ** num_qubits, dtype=np.int64)
    signs = jnp.zeros((2,) * num_qubits, dtype=np.uint8)

    for iq in range(num_qubits):
        signs = jax.lax.cond(
            jnp.equal(pauli_string[iq], 3),
            lambda: jnp.moveaxis(jnp.moveaxis(signs, iq, 0).at[1].add(1), 0, iq),
            lambda: signs
        )
    signs %= 2
    return rows, signs.reshape(-1)


@jax.jit
def _pauli_map_nondiagonal(pauli_string: jax.Array) -> tuple[jax.Array, jax.Array]:
    num_qubits = pauli_string.shape[0]
    rows = jnp.arange(2 ** num_qubits, dtype=np.int64).reshape((2,) * num_qubits)
    signs = jnp.zeros((2,) * num_qubits, dtype=np.uint8)

    for iq in range(num_qubits):
        rows = jax.lax.cond(
            jnp.not_equal(pauli_string[iq] % 3, 0),
            lambda: jnp.flip(rows, axis=iq),
            lambda: rows
        )
        signs = jax.lax.cond(
            jnp.greater(pauli_string[iq], 1),
            lambda: jnp.moveaxis(jnp.moveaxis(signs, iq, 0).at[1].add(1), 0, iq),
            lambda: signs
        )
    signs %= 2
    return rows.reshape(-1), signs.reshape(-1)


@jax.jit
def _subspace_pauli_map(
    pauli_string: jax.Array,
    states: jax.Array,
    subspace: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return the row numbers and matrix elements of the Pauli string for the given columns.

    Args:
        pauli_string: Shape [num_qubits]
        states: Shape [subspace_dim, num_qubits]
        subspace: Shape [subspace_dim, ceil(num_qubits / 8)]
    """
    return jax.lax.cond(
        jnp.all(jnp.equal(pauli_string % 3, 0)),
        lambda: _subspace_pauli_map_diagonal(pauli_string, states),
        lambda: _subspace_pauli_map_nondiagonal(pauli_string, states, subspace)
    )


_v_subspace_pauli_map = jax.jit(jax.vmap(_subspace_pauli_map, in_axes=(0, None, None)))


@jax.jit
def _sv_subspace_pauli_map(
    pauli_strings: jax.Array,
    states: jax.Array,
    subspace: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Temporally vectorized rows_and_elements. Cannot fullly vmap because of memory footprint."""
    def body_fn(carry, pauli_string_block):
        _states, _subspace = carry
        rows, signs = _v_subspace_pauli_map(pauli_string_block, _states, _subspace)
        return carry, (rows, signs)

    return jax.lax.scan(
        body_fn, (states, subspace), pauli_strings
    )[1]


_psv_subspace_pauli_map = jax.pmap(_sv_subspace_pauli_map, axis_name='device',
                                   in_axes=(0, None, None))


@jax.jit
def _subspace_pauli_map_diagonal(
    pauli_string: jax.Array,
    states: jax.Array
) -> tuple[jax.Array, jax.Array]:
    rows = jnp.arange(np.prod(states.shape[:-1]), dtype=np.int32).reshape(states.shape[:-1])
    signs = jnp.sum(states * jnp.equal(pauli_string, 3).astype(np.uint8), axis=-1) % 2
    return rows, signs


@jax.jit
def _subspace_pauli_map_nondiagonal(
    pauli_string: jax.Array,
    states: jax.Array,
    subspace: jax.Array
) -> tuple[jax.Array, jax.Array]:
    mapped_states = states ^ jnp.not_equal(pauli_string % 3, 0).astype(np.uint8)
    packed_mapped_states = jnp.packbits(mapped_states, axis=1)
    rows = subspace_indices(packed_mapped_states, subspace)
    signs = jnp.sum(states * jnp.greater(pauli_string, 1).astype(np.uint8), axis=-1) % 2
    return rows, signs


@jax.jit
def subspace_indices(states: jax.Array, subspace: jax.Array) -> jax.Array:
    """Return the positions of the states in the subspace, or -1 if not found."""
    # Borrowing from jax._src.lax.lax_numpy._searchsorted_via_sort
    def _rank(x):
        idx = jnp.arange(x.shape[0], dtype=np.int32)
        return jnp.empty_like(idx).at[jnp.lexsort(x.T[::-1])].set(idx)

    index = _rank(jnp.concatenate([states, subspace], axis=0))[:states.shape[0]]
    positions = (index - _rank(states)) % subspace.shape[0]
    return jnp.where(jnp.all(jnp.equal(states, subspace[positions]), axis=1), positions, -1)


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
