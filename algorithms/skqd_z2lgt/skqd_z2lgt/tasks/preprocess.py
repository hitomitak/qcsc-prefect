"""Preprocess raw data (link states with errors) and convert them to vertex and plaquette data."""
from collections.abc import Callable
import logging
from typing import Optional
import numpy as np
import h5py
from qiskit.primitives import BitArray
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.mwpm import convert_link_to_plaq, minimum_weight_link_state

RecoData = list[tuple[np.ndarray, np.ndarray]]  # [(vertex data, plaquette data)] * steps


def check_saved_reco(
    parameters: Parameters,
    output_filename: str,
    logger: Optional[logging.Logger] = None
) -> tuple[RecoData, RecoData] | None:
    logger = logger or logging.getLogger(__name__)

    num_steps = parameters.skqd.n_trotter_steps

    with h5py.File(output_filename, 'r') as source:
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


def save_reco(reco_data: tuple[RecoData, RecoData], output_filename: str):
    with h5py.File(output_filename, 'r+') as out:
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


def preprocess_flow(
    parameters: Parameters,
    bit_arrays: tuple[list[BitArray], list[BitArray]],
    output_filename: str,
    convert_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings.

    Args:
        parameters: Configuration parameters.
        cpu_pyfuncjob_name: Name of the PyFunctionJob block that runs a python function in an
            interpreter in the current environment.
        bit_arrays: Lists of BitArrays returned by sample_krylov_bitstrings.
        output_filename: Name of the HDF5 file where intermediate and final output of the workflow
            are written.
    """
    logger = logger or logging.getLogger(__name__)

    reco_data = check_saved_reco(parameters, output_filename, logger)
    if reco_data:
        return reco_data

    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    base_link_state = minimum_weight_link_state(parameters.lgt.charged_vertices, lattice)
    dual_lattice = lattice.plaquette_dual(base_link_state)

    reco_data = convert_fn(bit_arrays, dual_lattice)

    save_reco(reco_data, output_filename)

    return reco_data


def preprocess(
    parameters: Parameters,
    bit_arrays: tuple[list[BitArray], list[BitArray]],
    output_filename: str,
    logger: Optional[logging.Logger] = None
):
    def convert_fn(_bit_arrays, dual_lattice):
        reco_data = []
        for arrays in _bit_arrays:
            reco_data.append([])
            for array in arrays:
                reco_data[-1].append(convert_link_to_plaq(array, dual_lattice))
        return tuple(reco_data)

    return preprocess_flow(parameters, bit_arrays, output_filename, convert_fn, logger)
