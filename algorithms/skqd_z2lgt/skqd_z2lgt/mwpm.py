"""Functions for minimum-distance correction of observed link bitstrings."""
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import numpy as np
from scipy.sparse import csc_matrix
from pymatching import Matching
from qiskit.primitives import BitArray
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from heavyhex_qft.utils import as_bitarray
from heavyhex_qft.plaquette_dual import PlaquetteDual
from skqd_z2lgt.utils import read_bits


def make_matching(lattice: TriangularZ2Lattice) -> Matching:
    nv = lattice.num_vertices
    nl = lattice.num_links
    matching_matrix = np.zeros((nv, nl), dtype=int)
    for iv in range(lattice.num_vertices):
        matching_matrix[iv, lattice.vertex_links(iv)] = 1
    return Matching(csc_matrix(matching_matrix[::-1, ::-1]))


def mwpm_correct(
    link_state: np.ndarray,
    dual_lattice: TriangularZ2Lattice,
    matching: Optional[Matching] = None
) -> tuple[np.ndarray, np.ndarray]:
    if not matching:
        matching = make_matching(dual_lattice.primal)
    return _mwpm_correct(as_bitarray(link_state), dual_lattice, matching)


def _mwpm_correct(link_state, dual_lattice, matching):
    """Get the syndrome of the link state and apply the MWPM correction.

    When the base link state of the dual lattice is nonzero, we have two ways to correct:
    - Compute the syndrome, apply MWPM, XOR with the base state, correct
    - Compute the syndrome, XOR with the base syndrome, apply MWPM, correct
    Either way results in a unique (independent of the true link state) map from link bit-flip
    errors to Wilson loops, but the obtained loops will be different. For our purpose of learning
    the association between errors and loops (more specifically conditional probability of loops for
    given error syndromes), there is likely no real difference in the outcome between the two
    approaches. We choose the second approach ("neutralize" the syndrome) here. Possibly this allows
    reusing the generative model trained for circuits with different initial states.
    """
    syndrome = dual_lattice.primal.get_syndrome(link_state) ^ dual_lattice.base_syndrome
    correction = matching.decode(syndrome)
    corrected_link_state = link_state ^ correction
    return corrected_link_state, syndrome


def minimum_weight_link_state(
    charged_vertices: list[int],
    lattice: TriangularZ2Lattice,
    matching: Optional[Matching] = None
) -> np.ndarray:
    """Return a link state with minimal excitations corresponding to the charge distribution."""
    if matching is None:
        matching = make_matching(lattice)
    charge_config = np.zeros(lattice.num_vertices, dtype=int)
    charge_config[::-1][charged_vertices] = 1
    return matching.decode(charge_config)


def convert_link_to_plaq(
    bit_array: BitArray,
    dual_lattice: PlaquetteDual,
    shuffle: bool = False,
    batch_size: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the counts dict to input data for correction learning."""
    array = bit_array.array
    num_bits = bit_array.num_bits
    shots = array.shape[0]

    if batch_size <= 0:
        out = _batch_convert(array, num_bits, dual_lattice)
    else:
        with ProcessPoolExecutor() as executor:
            futures = []
            start = 0
            while start < shots:
                end = start + batch_size
                fut = executor.submit(_batch_convert, array[start:end], num_bits, dual_lattice)
                futures.append((start, end, fut))
                start = end
        out = (
            np.empty((shots, dual_lattice.primal.num_vertices), dtype=np.uint8),
            np.empty((shots, dual_lattice.num_plaquettes), dtype=np.uint8)
        )

        for start, end, fut in futures:
            batch_out = fut.result()
            out[0][start:end] = batch_out[0]
            out[1][start:end] = batch_out[1]

    if shuffle:
        indices = np.arange(shots)
        np.random.default_rng().shuffle(indices)
        out = (out[0][indices], out[1][indices])

    return out


def _batch_convert(batch_array, num_bits, dual_lattice):
    lattice = dual_lattice.primal
    matching = make_matching(lattice)
    out = (
        np.empty((batch_array.shape[0], lattice.num_vertices), dtype=np.uint8),
        np.empty((batch_array.shape[0], lattice.num_plaquettes), dtype=np.uint8)
    )

    link_states = read_bits(batch_array, num_bits=num_bits, align='right')

    for ishot, link_state in enumerate(link_states):
        corrected_link_state, syndrome = _mwpm_correct(link_state, dual_lattice, matching)
        plaquette_state = dual_lattice.map_link_state(corrected_link_state)
        out[0][ishot] = syndrome
        out[1][ishot] = plaquette_state

    return out
