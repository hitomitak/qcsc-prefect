"""Functions to pre- and postprocess data for correction learning."""
import numpy as np
from heavyhex_qft.plaquette_dual import PlaquetteDual
from .mwpm import make_matching, mwpm_correct


def preprocess(
    counts: dict[str, int] | tuple[np.ndarray, np.ndarray],
    dual_lattice: PlaquetteDual,
    as_counts: bool = False,
    shuffle: bool = True
):
    """Convert the counts dict to input data for correction learning."""
    lattice = dual_lattice._primal
    matching = make_matching(lattice)

    if isinstance(counts, dict):
        shots = sum(counts.values())
        generator = counts.items()
    else:
        shots = np.sum(counts[1])
        generator = zip(*counts)

    if as_counts:
        out = {}
    else:
        out = np.empty((shots, lattice.num_vertices + lattice.num_plaquettes), dtype=int)
    pos = 0
    for link_state, count in generator:
        link_state, syndrome = mwpm_correct(link_state, lattice, matching)
        plaquette_state = dual_lattice.map_link_state(link_state)
        if as_counts:
            out[(tuple(syndrome), tuple(plaquette_state))] = count
        else:
            out[pos:pos + count, :lattice.num_vertices] = syndrome[None, :]
            out[pos:pos + count, lattice.num_vertices:] = plaquette_state[None, :]
            pos += count

    if not as_counts and shuffle:
        np.random.shuffle(out)

    return out
