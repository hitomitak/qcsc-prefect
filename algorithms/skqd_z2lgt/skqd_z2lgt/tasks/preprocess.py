"""Preprocess raw data (link states with errors) and convert them to vertex and plaquette data."""
from collections.abc import Callable
import logging
from typing import Optional
import numpy as np
import h5py
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.mwpm import convert_link_to_plaq, minimum_weight_link_state

RecoData = list[tuple[np.ndarray, np.ndarray]]  # [(vertex data, plaquette data)] * steps


def check_saved_reco(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
) -> tuple[RecoData, RecoData] | None:
    logger = logger or logging.getLogger(__name__)

    num_steps = parameters.skqd.n_trotter_steps

    with h5py.File(parameters.output_filename, 'r', libver='latest') as source:
        data_group = source.get('data', {})
        if 'vtx' in data_group and 'plaq' in data_group:
            logger.info('Loading existing reco data from output file')
            return tuple(
                [
                    (data_group[f'vtx/{etype}_step{istep}'][()],
                     data_group[f'plaq/{etype}_step{istep}'][()])
                    for istep in range(num_steps)
                ]
                for etype in ['exp', 'ref']
            )

    return None


def save_reco(
    parameters: Parameters,
    reco_data: tuple[RecoData, RecoData],
    logger: Optional[logging.Logger] = None
):
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving vertex and plaquette data')
    with h5py.File(parameters.output_filename, 'r+', libver='latest') as out:
        data_group = out['data']
        groups = [data_group.get(gname) or data_group.create_group(gname)
                  for gname in ['vtx', 'plaq']]

        for etype, step_data in zip(['exp', 'ref'], reco_data):
            for istep, arrays in enumerate(step_data):
                dname = f'{etype}_step{istep}'
                for group, array in zip(groups, arrays):
                    try:
                        del group[dname]
                    except KeyError:
                        pass
                    dataset = group.create_dataset(dname, data=np.packbits(array, axis=1))
                    dataset.attrs['num_bits'] = array.shape[1]


def load_reco(
    parameters: Parameters,
    etype: Optional[str] = None,
    istep: Optional[int] = None
) -> tuple[RecoData, RecoData] | RecoData | tuple[np.ndarray, np.ndarray]:
    def read_bits(dataset):
        return np.unpackbits(dataset[()], axis=-1)[..., :dataset.attrs['num_bits']]

    if etype:
        etypes = [etype]
    else:
        etypes = ['exp', 'ref']
    if istep is not None:
        isteps = [istep]
    else:
        isteps = list(range(parameters.skqd.n_trotter_steps))

    with h5py.File(parameters.output_filename, 'r', libver='latest') as source:
        group = source['data']
        result = tuple(
            [(read_bits(group[f'vtx/{etype}_step{istep}']),
              read_bits(group[f'plaq/{etype}_step{istep}'])) for istep in isteps]
            for etype in etypes
        )
    if etype:
        result = result[0]
        if istep is not None:
            result = result[0]
    return result


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

    reco_data = check_saved_reco(parameters, logger)
    if reco_data:
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
