"""Functions related to MWPM."""
import numpy as np
from scipy.sparse import csc_matrix, csr_array
from pymatching import Matching
from heavyhex_qft.utils import as_bitarray


def make_matching(lattice):
    indices = []
    indptr = [0]
    for vertex in range(lattice.num_vertices):
        indices += lattice.vertex_links(vertex)
        indptr.append(len(indices))
    csr_graph = csr_array((np.ones(len(indices), dtype=int), indices, indptr),
                          shape=(lattice.num_vertices, lattice.num_links))
    return Matching(csc_matrix(csr_graph))


def mwpm_correct(link_state, lattice, matching=None):
    if not matching:
        matching = make_matching(lattice)

    link_state = as_bitarray(link_state)
    syndrome = lattice.get_syndrome(link_state)
    correction = matching.decode(syndrome).astype(bool)
    link_state[correction] = 1 - link_state[correction]

    return link_state, syndrome
