"""Common functions for skqd_z2lgt tasks."""
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from heavyhex_qft.plaquette_dual import PlaquetteDual
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.mwpm import minimum_weight_link_state

def make_dual_lattice(parameters: Parameters) -> PlaquetteDual:
    """Return the Ising hamiltonian for the given charge sector."""
    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    base_link_state = minimum_weight_link_state(parameters.lgt.charged_vertices, lattice)
    dual_lattice = lattice.plaquette_dual(base_link_state)
    return dual_lattice
