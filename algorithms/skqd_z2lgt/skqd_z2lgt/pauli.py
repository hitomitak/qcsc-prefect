# pylint: disable=cell-var-from-loop
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
    states: Optional[np.ndarray] = None
) -> BCOO:
    """Convert a SparsePauliOp to a sparse (COO) array, with optional subspace projection.

    Args:
        op: Sum of Pauli strings.
        states: List of bitstrings with shape [subspace_dim, num_qubits] or an integer array of
            indices of the computational basis in the full Hilbert space.

    Returns:
        A COO array encoding the op projected onto the subspace.
    """
    if states is None:
        proj_dim = 2 ** op.num_qubits
    else:
        states = np.asarray(states)
        if states.ndim == 1:
            # Convert integer indices to binary
            states = (states[:, None] >> np.arange(op.num_qubits)[None, ::-1]) % 2

        states = jnp.array(states)
        proj_dim = states.shape[0]

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

    if states is None:
        rows, phases = multi_pauli_map(pauli_strings)
    else:
        rows, phases = multi_subspace_pauli_map(pauli_strings, states)

    # Remove the device axis and truncate at the original number of op terms
    rows = rows[:num_terms].reshape(-1)
    cols = jnp.tile(jnp.arange(proj_dim), num_terms)
    data = ((1.j ** phases) * op_coeffs[:, None])[:num_terms].reshape(-1)

    if states is not None:
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


@jax.jit
def pauli_map(pauli_string: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the row indices and matrix element phases of nonzero entries of the Pauli unitary.

    Phases are integers in units of pi/2.

    This method only works for num_qubits < 64, but the practical limit for enumerating all rows is
    much lower.

    Args:
        pauli_string: Array of {0,1,2,3} pauli indices, shape [num_qubits].

    Returns:
        Row indices (shape [2 ** num_qubits]) and matrix element phases ([2 ** num_qubits]) of
        nonzero entries of the Pauli unitary, ordered by columns.
    """
    return jax.lax.cond(
        jnp.all(jnp.equal(pauli_string % 3, 0)),
        _pauli_map_diagonal,
        _pauli_map_nondiagonal,
        pauli_string
    )


v_pauli_map = jax.jit(jax.vmap(pauli_map))
pv_pauli_map = jax.pmap(v_pauli_map, axis_name='device')


def multi_pauli_map(pauli_strings: jax.Array) -> tuple[jax.Array, jax.Array]:
    match pauli_strings.ndim:
        case 2:
            rows, phases = v_pauli_map(pauli_strings)
        case 3:
            rows, phases = pv_pauli_map(pauli_strings)
            shape = (np.prod(pauli_strings.shape[:-1]), 2 ** pauli_strings.shape[-1])
            rows = rows.reshape(shape)
            phases = phases.reshape(shape)
        case _:
            raise ValueError('Too many dimensions in pauli_strings')

    return rows, phases


@jax.jit
def subspace_pauli_map(pauli_string: jax.Array, states: jax.Array) -> tuple[jax.Array, jax.Array]:
    packed_states = jnp.packbits(states, axis=-1)
    return _subspace_pauli_map(pauli_string, states, packed_states)


def multi_subspace_pauli_map(
    pauli_strings: jax.Array,
    states: jax.Array
) -> tuple[jax.Array, jax.Array]:
    packed_states = jnp.packbits(states, axis=-1)
    match pauli_strings.ndim:
        case 2:
            rows, phases = _v_subspace_pauli_map(pauli_strings, states, packed_states)
        case 3:
            rows, phases = _sv_subspace_pauli_map(pauli_strings, states, packed_states)
        case 4:
            rows, phases = _psv_subspace_pauli_map(pauli_strings, states, packed_states)
        case _:
            raise ValueError('Too many dimensions in pauli_strings')

    shape = (np.prod(pauli_strings.shape[:-1]), np.prod(states.shape[:-1]))
    rows = rows.reshape(shape)
    phases = phases.reshape(shape)

    return rows, phases


@jax.jit
def _pauli_map_diagonal(pauli_string: jax.Array) -> jax.Array:
    num_qubits = pauli_string.shape[0]
    rows = jnp.arange(2 ** num_qubits, dtype=np.int64)
    phases = jnp.zeros((2,) * num_qubits, dtype=np.uint8)

    for iq in range(num_qubits):
        phases = jax.lax.cond(
            jnp.equal(pauli_string[iq], 3),
            lambda: jnp.moveaxis(jnp.moveaxis(phases, iq, 0).at[1].add(2), 0, iq),
            lambda: phases
        )
    phases %= 4
    return rows, phases.reshape(-1)


@jax.jit
def _pauli_map_nondiagonal(pauli_string: jax.Array) -> jax.Array:
    num_qubits = pauli_string.shape[0]
    rows = jnp.arange(2 ** num_qubits, dtype=np.int64).reshape((2,) * num_qubits)
    phases = jnp.zeros((2,) * num_qubits, dtype=np.uint8)

    for iq in range(num_qubits):
        rows = jax.lax.cond(
            jnp.not_equal(pauli_string[iq] % 3, 0),
            lambda: jnp.flip(rows, axis=iq),
            lambda: rows
        )
        phases += jnp.equal(pauli_string[iq], 2).astype(np.uint8)
        phases = jax.lax.cond(
            jnp.greater(pauli_string[iq], 1),
            lambda: jnp.moveaxis(jnp.moveaxis(phases, iq, 0).at[1].add(2), 0, iq),
            lambda: phases
        )
    phases %= 4

    return rows.reshape(-1), phases.reshape(-1)


@jax.jit
def _subspace_pauli_map(
    pauli_string: jax.Array,
    states: jax.Array,
    packed_states: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return the row numbers and matrix elements of the Pauli string for the given columns.

    Args:
        pauli_string: Shape [num_qubits]
        states: Shape [subspace_dim, num_qubits]
        packed_states: Shape [subspace_dim, ceil(num_qubits / 8)]
    """
    return jax.lax.cond(
        jnp.all(jnp.equal(pauli_string % 3, 0)),
        lambda: _subspace_pauli_map_diagonal(pauli_string, states),
        lambda: _subspace_pauli_map_nondiagonal(pauli_string, states, packed_states)
    )


_v_subspace_pauli_map = jax.jit(jax.vmap(_subspace_pauli_map, in_axes=(0, None, None)))


@jax.jit
def _sv_subspace_pauli_map(
    pauli_strings: jax.Array,
    indices: jax.Array,
    packed_indices: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Temporally vectorized rows_and_elements. Cannot fullly vmap because of memory footprint."""
    def body_fn(carry, pauli_string_block):
        _indices, _packed_indices = carry
        rows, phases = _v_subspace_pauli_map(pauli_string_block, _indices, _packed_indices)
        return carry, (rows, phases)

    return jax.lax.scan(
        body_fn, (indices, packed_indices), pauli_strings
    )[1]


_psv_subspace_pauli_map = jax.pmap(_sv_subspace_pauli_map, axis_name='device',
                                   in_axes=(0, None, None))


@jax.jit
def _subspace_pauli_map_diagonal(
    pauli_string: jax.Array,
    states: jax.Array
) -> jax.Array:
    rows = jnp.arange(np.prod(states.shape[:-1]), dtype=np.int64).reshape(states.shape[:-1])
    phases = jnp.sum(states * jnp.equal(pauli_string, 3).astype(np.uint8), axis=-1) * 2
    phases %= 4
    return rows, phases


@jax.jit
def _subspace_pauli_map_nondiagonal(
    pauli_string: jax.Array,
    states: jax.Array,
    packed_states: jax.Array
) -> jax.Array:
    mapped_states = states ^ jnp.not_equal(pauli_string % 3, 0).astype(np.uint8)
    packed_mapped_states = jnp.packbits(mapped_states, axis=-1)
    if packed_mapped_states.ndim == 2:
        packed_mapped_states = packed_mapped_states.reshape((1,) + packed_mapped_states.shape)
    rows = subspace_indices(packed_mapped_states, packed_states)
    phases = jnp.sum(states * jnp.greater(pauli_string, 1).astype(np.uint8), axis=-1) * 2
    phases += jnp.sum(jnp.equal(pauli_string, 2).astype(np.uint8))
    phases %= 4
    return rows, phases


@jax.jit
def subspace_index(packed_state: jax.Array, pool: jax.Array) -> int:
    """Return the position of the bitstring in the pool, or -1 if not found."""
    matches = jnp.all(jnp.equal(packed_state, pool), axis=1)
    idx = jnp.argmax(matches)
    return jax.lax.select(jnp.equal(idx, 0), jax.lax.select(matches[0], 0, -1), idx)


v_subspace_index = jax.jit(jax.vmap(subspace_index, in_axes=(0, None)))
"""Bitstring position finder vectorized on bitstrings."""


@jax.jit
def subspace_indices(packed_states: jax.Array, pool: jax.Array) -> jax.Array:
    """Build an array of index blocks with shape (num_blocks, block_size).

    Args:
        packed_states: Shape [num_block, block_size, num_qubits]
        pool: Shape [pool_size, num_qubits]

    Returns:
        Index array of shape [num_block * block_size].
    """
    def body_fn(_pool, bitstring):
        indices = v_subspace_index(bitstring, _pool)
        return _pool, indices

    return jax.lax.scan(
        body_fn, pool, packed_states
    )[1].reshape((-1,))


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
