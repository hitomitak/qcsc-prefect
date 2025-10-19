"""Functions to pre- and postprocess data for correction learning."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from heavyhex_qft.plaquette_dual import PlaquetteDual
from skqd_z2lgt.mwpm import make_matching, mwpm_correct


def preprocess(
    counts: dict[str, int] | tuple[np.ndarray, np.ndarray],
    dual_lattice: PlaquetteDual,
    as_counts: bool = False,
    shuffle: bool = True,
    batch_size: int = 0
):
    """Convert the counts dict to input data for correction learning."""
    if isinstance(counts, dict):
        shots = sum(counts.values())
        generator = counts.items()
    else:
        shots = np.sum(counts[1])
        generator = zip(*counts)

    if batch_size <= 0:
        out = _batch_process(generator, dual_lattice, as_counts, shots=shots)
    else:
        with ProcessPoolExecutor() as executor:
            geniter = iter(generator)
            futures = []
            while True:
                batch = []
                try:
                    for _ in range(batch_size):
                        batch.append(next(geniter))
                except StopIteration:
                    break
                finally:
                    fut = executor.submit(_batch_process, batch, dual_lattice, as_counts)
                    futures.append(fut)

        if as_counts:
            out = {}
        else:
            out = (
                np.empty((shots, dual_lattice.primal.num_vertices), dtype=np.uint8),
                np.empty((shots, dual_lattice.num_plaquettes), dtype=np.uint8)
            )

        pos = 0
        for fut in futures:
            batch_out = fut.result()
            if as_counts:
                out.update(batch_out)
            else:
                batch_counts = batch_out[0].shape[0]
                out[0][pos:pos + batch_counts] = batch_out[0]
                out[1][pos:pos + batch_counts] = batch_out[1]
                pos += batch_counts

    if not as_counts and shuffle:
        indices = np.arange(pos)
        np.random.default_rng().shuffle(indices)
        out = (out[0][indices], out[1][indices])

    return out


def _batch_process(batch, dual_lattice, as_counts, shots=None):
    lattice = dual_lattice.primal
    matching = make_matching(lattice)
    if as_counts:
        out = {}
    else:
        if shots is None:
            shots = sum((cnt for _, cnt in batch))
        out = (
            np.empty((shots, lattice.num_vertices), dtype=np.uint8),
            np.empty((shots, lattice.num_plaquettes), dtype=np.uint8)
        )

    pos = 0
    for link_state, count in batch:
        link_state, syndrome = mwpm_correct(link_state, lattice, matching)
        plaquette_state = dual_lattice.map_link_state(link_state)
        if as_counts:
            out[(tuple(syndrome.tolist()), tuple(plaquette_state.tolist()))] = int(count)
        else:
            out[0][pos:pos + count] = syndrome[None, :]
            out[1][pos:pos + count] = plaquette_state[None, :]
            pos += count

    return out
