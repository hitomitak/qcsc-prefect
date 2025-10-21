"""Process the link-state bitstrings with MWPM."""
import argparse
import logging
import time
from typing import Optional
import numpy as np
import h5py
from qiskit.primitives import BitArray
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.recovery_learning import preprocess

LOG = logging.getLogger(__name__)


def main(
    filename: str,
    result_index: Optional[int | list[int]] = None,
    out_filename: Optional[str] = None
):
    with h5py.File(filename, 'r', swmr=True) as source:
        configuration = dict(source.attrs)

        if result_index is None:
            result_index = list(range(2 * configuration['num_steps']))
        elif isinstance(result_index, int):
            result_index = [result_index]

        bit_arrays = {}
        for idx in result_index:
            if idx < configuration['num_steps']:
                etype = 'exp'
            else:
                etype = 'ref'
            istep = idx % configuration['num_steps']
            dataset = source[f'data/raw/{etype}_step{istep}']
            bit_arrays[idx] = BitArray(dataset[()], int(dataset.attrs['num_bits']))

    start = time.time()

    lattice = TriangularZ2Lattice(configuration['lattice'])
    dual_lattice = lattice.plaquette_dual()

    preprocessed = {}
    for idx, bit_array in bit_arrays.items():
        LOG.info('Processing BitArray from experiment %d', idx)
        preprocessed[idx] = preprocess(bit_array, dual_lattice, batch_size=4000)

    LOG.info('State conversion took %.2f seconds.', time.time() - start)

    file_mode = 'w' if out_filename else 'r+'
    out_filename = out_filename or filename

    with h5py.File(out_filename, file_mode) as out:
        lengths = [lattice.num_vertices, lattice.num_plaquettes]
        data_group = out['data']
        groups = [data_group.get(gname) or data_group.create_group(gname)
                  for gname in ['vtx', 'plaq']]

        for idx, arrays in preprocessed.items():
            if idx < configuration['num_steps']:
                etype = 'exp'
            else:
                etype = 'ref'
            istep = idx % configuration['num_steps']
            dname = f'{etype}_step{istep}'
            for group, array, num_bits in zip(groups, arrays, lengths):
                try:
                    del group[dname]
                except KeyError:
                    pass
                dataset = group.create_dataset(dname, data=np.packbits(array, axis=1))
                dataset.attrs['num_bits'] = num_bits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--result-index', nargs='+', type=int)
    parser.add_argument('--out-filename')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    main(options.filename, options.result_index, out_filename=options.out_filename)
