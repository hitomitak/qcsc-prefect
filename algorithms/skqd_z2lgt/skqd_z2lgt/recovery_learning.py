"""Functions to pre- and postprocess data for correction learning."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from qiskit.primitives import BitArray
from heavyhex_qft.plaquette_dual import PlaquetteDual
from skqd_z2lgt.mwpm import make_matching, mwpm_correct


def preprocess(
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
        out = _batch_process(array, num_bits, dual_lattice)
    else:
        with ProcessPoolExecutor() as executor:
            futures = []
            start = 0
            while start < shots:
                end = start + batch_size
                fut = executor.submit(_batch_process, array[start:end], num_bits, dual_lattice)
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


def _batch_process(batch_array, num_bits, dual_lattice):
    lattice = dual_lattice.primal
    matching = make_matching(lattice)
    out = (
        np.empty((batch_array.shape[0], lattice.num_vertices), dtype=np.uint8),
        np.empty((batch_array.shape[0], lattice.num_plaquettes), dtype=np.uint8)
    )

    # heavyhex_qft uses a little endian convention for some reason -> ::-1
    link_states = np.unpackbits(batch_array, axis=1)[:, -num_bits:][:, ::-1]

    for ishot, link_state in enumerate(link_states):
        corrected_link_state, syndrome = mwpm_correct(link_state, lattice, matching)
        plaquette_state = dual_lattice.map_link_state(corrected_link_state)
        out[0][ishot] = syndrome
        out[1][ishot] = plaquette_state

    return out
