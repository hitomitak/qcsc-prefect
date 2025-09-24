"""Sample-based Krylov quantum diagonalization."""
from collections.abc import Iterable
import logging
import time
import numpy as np
from scipy.sparse.linalg import eigsh
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, bcoo_dot_general
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_sqd.qubit import sort_and_remove_duplicates, project_operator_to_subspace
from skqd_z2lgt.jax_experimental_sparse_linalg import lobpcg_standard
from skqd_z2lgt.pauli import to_bcoo

LOG = logging.getLogger(__name__)


def keys_to_intset(keys: Iterable[str]) -> set:
    keys_arr = np.array([list(map(int, key)) for key in keys])
    return set(np.sum(keys_arr * (1 << np.arange(keys_arr.shape[1])[::-1]), axis=1).tolist())


def sqd(hamiltonian: SparsePauliOp, indices: np.ndarray, device_id: int):
    LOG.info('%d configurations', indices.shape[0])

    with jax.default_device(jax.devices()[device_id]):
        start = time.time()
        hproj = to_bcoo(hamiltonian, indices)
        end = time.time()
        LOG.info('%f seconds to project', end - start)
        start = end
        ground_energy, _ = ground_state_lobpcg(hproj)
        end = time.time()
        LOG.info('%f seconds to diagonalize', end - start)

    return ground_energy


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


def qiskit_sqd(bitstrings, hamiltonian, jax_device_id=None):
    bitstring_matrix = np.unique(bitstrings, axis=0)[:, ::-1].astype(bool)
    if jax_device_id is None:
        device = jax.default_device()
    else:
        device = jax.devices()[jax_device_id]

    with jax.default_device(device):
        bitstring_matrix = sort_and_remove_duplicates(bitstring_matrix)
        ham_proj = project_operator_to_subspace(bitstring_matrix, hamiltonian)

    evals, evecs = eigsh(ham_proj, k=1, which='SA')
    return bitstring_matrix, ham_proj, evals[0], evecs[:, 0]
