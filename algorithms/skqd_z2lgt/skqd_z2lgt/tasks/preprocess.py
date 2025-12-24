# pylint: disable=invalid-name
"""Preprocess raw data (link states with errors) and convert them to vertex and plaquette data."""
import os
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import h5py
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.mwpm import convert_link_to_plaq
from skqd_z2lgt.utils import read_bits, save_bits
from skqd_z2lgt.tasks.sample_quantum import load_raw
from skqd_z2lgt.tasks.common import make_dual_lattice

RecoData = list[tuple[np.ndarray, np.ndarray]]  # [(vertex data, plaquette data)] * steps


def save_reco(
    parameters: Parameters,
    reco_data: tuple[RecoData, RecoData],
    logger: Optional[logging.Logger] = None
):
    """Save vertex and plaquette data to files."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving vertex and plaquette data')
    dirpath = Path(parameters.pkgpath) / 'data' / 'reco'
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    for etype, arrays in zip(['exp', 'ref'], reco_data):
        for istep in range(parameters.skqd.num_krylov):
            path = dirpath / f'{etype}_step{istep}.h5'
            with h5py.File(path, 'w', libver='latest') as out:
                for name, array in zip(['vtx', 'plaq'], arrays[istep]):
                    save_bits(out, name, array)


def load_reco(
    parameters: Parameters,
    etype: Optional[str] = None,
    istep: Optional[int] = None
) -> tuple[RecoData, RecoData] | RecoData | tuple[np.ndarray, np.ndarray]:
    """Load vertex and plaquette data from files."""
    if etype is None:
        etypes = ['exp', 'ref']
    else:
        etypes = [etype]
    if istep is None:
        isteps = list(range(parameters.skqd.num_krylov))
    else:
        isteps = [istep]

    dirpath = Path(parameters.pkgpath) / 'data' / 'reco'
    data = tuple([] for _ in range(len(etypes)))
    for etp, rdata in zip(etypes, data):
        for istp in isteps:
            with h5py.File(dirpath / f'{etp}_step{istp}.h5', 'r', libver='latest') as source:
                rdata.append((read_bits(source['vtx']), read_bits(source['plaq'])))

    if etype is None:
        return data
    if istep is None:
        return data[0]
    return data[0][0]


def preprocess_flow(
    parameters: Parameters,
    convert_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings."""
    logger = logger or logging.getLogger(__name__)

    try:
        load_reco(parameters)
    except FileNotFoundError:
        pass
    else:
        logger.info('All link-state bitstrings already converted')
        return

    logger.info('Correcting the charge sector of link-state bitstrings and converting them to '
                'vertex and plaquette data')
    convert_fn()


def preprocess(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
):
    """Standalone preprocess function."""
    def convert_fn():
        dual_lattice = make_dual_lattice(parameters)
        raw_data = load_raw(parameters)

        reco_data = ([], [])
        batch_size = parameters.runtime.shots // 20
        for raw, reco in zip(raw_data, reco_data):
            for array in raw:
                reco.append(convert_link_to_plaq(array, dual_lattice, batch_size=batch_size))

        save_reco(parameters, reco_data)

    preprocess_flow(parameters, convert_fn, logger=logger)


def preprocess_single_array(
    parameters: Parameters,
    etype: str,
    istep: int,
    logger: Optional[logging.Logger] = None
):
    """Preprocess one bit array parallelized over all cores and save the result."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Preprocessing %s step %d', etype, istep)

    bit_array = load_raw(parameters, etype, istep)
    dual_lattice = make_dual_lattice(parameters)
    batch_size = bit_array.num_shots // (os.cpu_count() - 1)
    arrays = convert_link_to_plaq(bit_array, dual_lattice, batch_size)

    path = Path(parameters.pkgpath) / 'data' / 'reco' / f'{etype}_step{istep}.h5'
    try:
        os.makedirs(path.parent)
    except FileExistsError:
        pass

    with h5py.File(path, 'w', libver='latest') as out:
        for name, array in zip(['vtx', 'plaq'], arrays):
            save_bits(out, name, array)


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--etype')
    parser.add_argument('--istep', type=int)
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()),
                        stream=sys.stdout)

    with open(Path(options.pkgpath) / 'parameters.json', 'r', encoding='utf-8') as src:
        params = Parameters.model_validate_json(src.read())

    if options.mpi:
        from mpi4py import MPI  # pylint: disable=no-name-in-module
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        nsteps = params.skqd.num_krylov
        if mpi_rank < nsteps:
            et = 'exp'
        else:
            et = 'ref'
        ist = mpi_rank % nsteps
    else:
        et = options.etype
        ist = options.istep

    preprocess_single_array(params, et, ist)
