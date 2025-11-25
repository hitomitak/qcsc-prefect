"""Subspace extension functions."""
import math
import numpy as np
import rustworkx as rx
from heavyhex_qft.plaquette_dual import PlaquetteDual
from heavyhex_qft.pure_z2_lgt import DummyPlaquette


def perturbation_0q(states: np.ndarray, dual_lattice: PlaquetteDual):
    """Append perturbative states up to 4th order in the 0-charge sector."""
    return np.concatenate([states, _plaquette_excitations(dual_lattice)], axis=0)


def perturbation_2q(states: np.ndarray, dual_lattice: PlaquetteDual):
    """Append perturbative states in a 2-charge sector."""
    lattice = dual_lattice.primal
    charge_dist = lattice.get_syndrome(dual_lattice.base_link_state)
    source, target = np.nonzero(charge_dist[::-1])[0]
    paths = rx.all_shortest_paths(lattice.graph, source, target)
    nodes_to_edge = {}
    for link_id, (n1, n2, _) in lattice.graph.edge_index_map().items():
        nodes_to_edge[(n1, n2)] = link_id
    free_grounds = np.empty((len(paths), lattice.num_plaquettes), dtype=np.uint8)
    for ipath, path in enumerate(paths):
        rev_link_state = np.zeros(lattice.num_links, dtype=np.uint8)
        links = [nodes_to_edge[(n1, n2)] for n1, n2 in zip(path[:-1], path[1:])]
        rev_link_state[links] = 1
        free_grounds[ipath] = dual_lattice.map_link_state(rev_link_state[::-1])
    excitations = _plaquette_excitations(dual_lattice)
    free_grounds = np.tile(free_grounds, (1, excitations.shape[0]))
    pert_states = free_grounds.reshape((len(paths),) + excitations.shape) ^ excitations
    pert_states = np.unique(pert_states.reshape((-1, states.shape[1])), axis=0)
    return np.concatenate([states, pert_states], axis=0)


def _plaquette_excitations(dual_lattice: PlaquetteDual):
    num_p = dual_lattice.num_plaquettes
    num_states = sum(math.comb(num_p, i) for i in range(5))
    states = np.zeros((num_states, num_p), dtype=np.uint8)
    # 1 plaquette
    start = 1
    end = start + num_p
    states[np.arange(start, end), np.arange(num_p)[::-1]] = 1
    start = end
    # 2 plaquettes
    for idx in range(num_p):
        indices = np.arange(idx + 1, num_p)
        end = start + indices.shape[0]
        states[np.arange(start, end), idx] = 1
        states[np.arange(start, end), indices] = 1
        start = end
    # 3 plaquettes
    for idx1 in range(num_p):
        for idx2 in range(idx1 + 1, num_p):
            indices = np.arange(idx2 + 1, num_p)
            end = start + indices.shape[0]
            states[np.arange(start, end), idx1] = 1
            states[np.arange(start, end), idx2] = 1
            states[np.arange(start, end), indices] = 1
            start = end
    # 4 plaquettes
    for idx1 in range(num_p):
        for idx2 in range(idx1 + 1, num_p):
            for idx3 in range(idx2 + 1, num_p):
                indices = np.arange(idx3 + 1, num_p)
                end = start + indices.shape[0]
                states[np.arange(start, end), idx1] = 1
                states[np.arange(start, end), idx2] = 1
                states[np.arange(start, end), idx3] = 1
                states[np.arange(start, end), indices] = 1
                start = end

    return states


def vertical_reflection(states: np.ndarray, dual_lattice: PlaquetteDual):
    """Append a vertically reflected state for every entry in states.

    For this operation to make sense, the lattice must have an even number of plaquettes that can
    be divided into upper and lower halves, and the two subgraphs formed by those halves must be
    isomorphic to each other.
    """
    dual_graph = dual_lattice.graph
    num_p = dual_lattice.num_plaquettes
    upper = dual_graph.subgraph(list(range(num_p // 2)))
    lower = dual_graph.subgraph(list(range(num_p // 2, num_p)))
    boundary_matches = {}
    boundary = num_p // 2 - 0.5
    for n1, n2, _ in dual_graph.edge_index_map().values():
        plaq1 = dual_graph[n1]
        plaq2 = dual_graph[n2]
        if isinstance(plaq1, DummyPlaquette) or isinstance(plaq2, DummyPlaquette):
            continue
        if (((p1 := dual_graph[n1].plaq_id) - boundary)
                * ((p2 := dual_graph[n2].plaq_id) - boundary)) < 0:
            boundary_matches[p1] = p2
            boundary_matches[p2] = p1

    def node_matcher(plaq1, plaq2):
        if (match := boundary_matches.get(plaq1.plaq_id)) is None:
            return True
        return plaq2.plaq_id == match

    vf2 = list(rx.vf2_mapping(upper, lower, node_matcher=node_matcher))[0]
    keys = [upper[key].plaq_id for key in vf2.keys()]
    values = [lower[value].plaq_id for value in vf2.values()]
    reflection = np.empty(num_p, dtype=np.int32)
    reflection[keys] = values
    reflection[values] = keys
    reflection = num_p - 1 - reflection[::-1]
    return np.concatenate([states, states[:, reflection]], axis=0)


extensions = {
    'perturbation_0q': perturbation_0q,
    'perturbation_2q': perturbation_2q,
    'vertical_reflection': vertical_reflection
}
