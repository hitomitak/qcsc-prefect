"""Preprocess raw data (link states with errors) and convert them to vertex and plaquette data."""
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import h5py
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.mwpm import convert_link_to_plaq, minimum_weight_link_state
from skqd_z2lgt.utils import read_bits, save_bits

RecoData = list[tuple[np.ndarray, np.ndarray]]  # [(vertex data, plaquette data)] * steps


def save_reco(
    parameters: Parameters,
    reco_data: tuple[RecoData, RecoData],
    logger: Optional[logging.Logger] = None
):
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving vertex and plaquette data')
    for igroup, group in enumerate(['vtx', 'plaq']):
        path = Path(parameters.output_filename) / 'data' / f'{group}.h5'
        with h5py.File(path, 'w', libver='latest') as out:
            for etype, step_data in zip(['exp', 'ref'], reco_data):
                group = out.create_group(etype)
                for istep, arrays in enumerate(step_data):
                    dname = f'step{istep}'
                    save_bits(group, dname, arrays[igroup])


def load_reco(
    parameters: Parameters,
    etype: Optional[str] = None,
    istep: Optional[int] = None
) -> tuple[RecoData, RecoData] | RecoData | tuple[np.ndarray, np.ndarray]:
    if etype:
        etypes = [etype]
    else:
        etypes = ['exp', 'ref']
    if istep is not None:
        isteps = [istep]
    else:
        isteps = list(range(parameters.skqd.n_trotter_steps))

    group_data = {}
    for group in ['vtx', 'plaq']:
        path = Path(parameters.output_filename) / 'data' / f'{group}.h5'
        with h5py.File(path, 'r', libver='latest', swmr=True) as source:
            group_data[group] = {
                et: {ist: read_bits(source[f'{et}/step{ist}']) for ist in isteps}
                for et in etypes
            }

    if etype:
        if istep is not None:
            return (group_data['vtx'][etype][istep], group_data['plaq'][etype][istep])
        return [(group_data['vtx'][etype][ist], group_data['plaq'][etype][ist])
                for ist in isteps]
    return tuple(
        [(group_data['vtx'][et][ist], group_data['plaq'][et][ist]) for ist in isteps]
        for et in etypes
    )


def preprocess_flow(
    parameters: Parameters,
    raw_data: tuple[RecoData, RecoData],
    convert_fn: Callable,
    logger: Optional[logging.Logger] = None
) -> tuple[RecoData, RecoData]:
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings.

    Args:
        parameters: Configuration parameters.
        cpu_pyfuncjob_name: Name of the PyFunctionJob block that runs a python function in an
            interpreter in the current environment.
        bit_arrays: Lists of BitArrays returned by sample_krylov_bitstrings.
    """
    logger = logger or logging.getLogger(__name__)

    try:
        reco_data = load_reco(parameters)
    except FileNotFoundError:
        pass
    else:
        logger.info('Loading existing reco data from output file')
        return reco_data

    logger.info('Correcting the charge sector of link-state bitstrings and converting them to '
                'vertex and plaquette data')

    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    base_link_state = minimum_weight_link_state(parameters.lgt.charged_vertices, lattice)
    dual_lattice = lattice.plaquette_dual(base_link_state)

    reco_data = convert_fn(raw_data, dual_lattice)

    save_reco(parameters, reco_data)

    return reco_data


def preprocess(
    parameters: Parameters,
    raw_data: tuple[RecoData, RecoData],
    logger: Optional[logging.Logger] = None
) -> tuple[RecoData, RecoData]:
    def convert_fn(bit_arrays, dual_lattice):
        batch_size = parameters.runtime.shots // 20
        reco_data = []
        for arrays in bit_arrays:
            reco_data.append([])
            for array in arrays:
                reco_data[-1].append(convert_link_to_plaq(array, dual_lattice,
                                                          batch_size=batch_size))
        return tuple(reco_data)

    return preprocess_flow(parameters, raw_data, convert_fn, logger)
