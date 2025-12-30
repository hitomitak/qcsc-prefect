# pylint: disable=invalid-name
"""Preprocess raw data (link states with errors) and convert them to vertex and plaquette data."""
import os
from collections.abc import Callable
import logging
from pathlib import Path
from itertools import product
from typing import Optional
import numpy as np
import h5py
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.mwpm import convert_link_to_plaq
from skqd_z2lgt.utils import read_bits, save_bits
from skqd_z2lgt.tasks.sample_quantum import load_raw
from skqd_z2lgt.tasks.common import make_dual_lattice

RecoData = tuple[np.ndarray, np.ndarray]  # (vertex data, plaquette data)


def save_reco(
    parameters: Parameters,
    arrays: RecoData,
    etype: str,
    idt: int,
    ikrylov: int,
    logger: Optional[logging.Logger] = None
):
    """Save vertex and plaquette data to files."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving reco data for etype=%s idt=%d ikrylov=%d', etype, idt, ikrylov)

    path = Path(parameters.pkgpath) / 'data' / 'reco' / f'{etype}_dt{idt}_k{ikrylov}.h5'
    os.makedirs(path.parent, exist_ok=True)
    with h5py.File(path, 'w', libver='latest') as out:
        for name, array in zip(['vtx', 'plaq'], arrays):
            save_bits(out, name, array)


def check_reco(
    parameters: Parameters,
    etype: str,
    idt: int,
    ikrylov: int,
    logger: Optional[logging.Logger] = None
) -> bool:
    logger = logger or logging.getLogger(__name__)
    path = Path(parameters.pkgpath) / 'data' / 'reco' / f'{etype}_dt{idt}_k{ikrylov}.h5'
    if os.path.exists(path):
        logger.info('Reco data for etype=%s idt=%d ikrylov=%d already exists.', etype, idt, ikrylov)
        return True
    return False


def load_reco(
    parameters: Parameters,
    etype: Optional[str] = None,
    idt: Optional[int] = None,
    ikrylov: Optional[int] = None
) -> (tuple[list[list[RecoData]], list[list[RecoData]]] | list[list[RecoData]] | list[RecoData]
      | RecoData):
    """Load vertex and plaquette data from files."""
    def read_reco_data(et, itm, ikr):
        path = Path(parameters.pkgpath) / 'data' / 'reco' / f'{et}_dt{itm}_k{ikr}.h5'
        with h5py.File(path, 'r', libver='latest') as source:
            return tuple(read_bits(source[name]) for name in ['vtx', 'plaq'])

    def read_dt_data(et, itm):
        return [read_reco_data(et, itm, ikr) for ikr in range(1, parameters.skqd.num_krylov + 1)]

    def read_et_data(et):
        return [read_dt_data(et, itm) for itm in range(len(parameters.skqd.time_steps))]

    if etype is None:
        return tuple(read_et_data(et) for et in ['exp', 'ref'])
    if idt is None:
        return read_et_data(etype)
    if ikrylov is None:
        return read_dt_data(etype, idt)
    return read_reco_data(etype, idt, ikrylov)


def preprocess_flow(
    parameters: Parameters,
    convert_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """Correct the link-state bitstrings with MWPM and convert to plaquette-state bitstrings."""
    logger = logger or logging.getLogger(__name__)

    tasks = []
    for etype in ['exp', 'ref']:
        for idt in range(len(parameters.skqd.time_steps)):
            for ikrylov in range(1, parameters.skqd.num_krylov + 1):
                if not check_reco(parameters, etype, idt, ikrylov, logger=logger):
                    tasks.append((etype, idt, ikrylov))

    if not tasks:
        logger.info('All link-state bitstrings already converted to vertex and plaquette data')
        return

    logger.info('Correcting %d raw BitArrays to vertex and plaquette data', len(tasks))
    convert_fn(tasks)


def preprocess(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
):
    """Standalone preprocess function."""
    def convert_fn(tasks):
        for etype, idt, ikrylov in tasks:
            preprocess_single_array(parameters, etype, idt, ikrylov, logger=logger)

    preprocess_flow(parameters, convert_fn, logger=logger)


def preprocess_single_array(
    parameters: Parameters,
    etype: str,
    idt: int,
    ikrylov: int,
    logger: Optional[logging.Logger] = None
):
    """Preprocess one bit array parallelized over all cores and save the result."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Preprocessing %s, time interval %f, Krylov vector %d',
                etype, parameters.skqd.time_steps[idt], ikrylov)

    dual_lattice = make_dual_lattice(parameters)
    batch_size = getattr(parameters.runtime, f'shots_{etype}') // (os.cpu_count() - 1)
    bit_array = load_raw(parameters, etype, idt, ikrylov)
    arrays = convert_link_to_plaq(bit_array, dual_lattice, batch_size)
    save_reco(parameters, arrays, etype, idt, ikrylov, logger=logger)


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--etype')
    parser.add_argument('--idt')
    parser.add_argument('--ikrylov')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()),
                        stream=sys.stdout)

    with open(Path(options.pkgpath) / 'parameters.json', 'r', encoding='utf-8') as src:
        params = Parameters.model_validate_json(src.read())

    if options.etype:
        etypes = options.etype.split(',')
        idts = list(map(int, options.idt.split(',')))
        ikrylovs = list(map(int, options.ikrylov.split(',')))
        if not len(etypes) == len(idts) == len(ikrylovs):
            raise ValueError('Lengths of etype, idt, and ikrylov lists do not match')
        task_specs = list(zip(etypes, idts, ikrylovs))
    else:
        etypes = ['exp', 'ref']
        idts = list(range(len(params.skqd.time_steps)))
        ikrylovs = list(range(1, params.skqd.num_krylov + 1))
        task_specs = list(product(etypes, idts, ikrylovs))

    if options.mpi:
        from mpi4py import MPI  # pylint: disable=no-name-in-module
        comm = MPI.COMM_WORLD
        task_specs = [task_specs[comm.Get_rank()]]

    for task_spec in task_specs:
        preprocess_single_array(params, *task_spec)
