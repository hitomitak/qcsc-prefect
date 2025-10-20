"""Process the link-state bitstrings with MWPM."""
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import numpy as np
import h5py
from qiskit_ibm_runtime import QiskitRuntimeService
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.recovery_learning import preprocess

LOG = logging.getLogger(__name__)


def main(filename: str, result_index: int | list[int], out_filename: Optional[str] = None):
    with h5py.File(filename, 'r', swmr=True) as source:
        configuration = {}
        for key in source['configuration'].keys():
            record = source[f'configuration/{key}'][()]
            if isinstance(record, bytes):
                record = record.decode()
            configuration[key] = record

        job_id = source['experiment/job_id'][()].decode()  # pylint: disable=no-member

    LOG.info('Job ID: %s', job_id)

    service = QiskitRuntimeService(instance=configuration['instance'])
    job_result = service.job(job_id).result()

    LOG.info('Retrieved results of %d PUBs. Converting link states to vertex and plaquette states.',
             len(job_result))
    start = time.time()

    lattice = TriangularZ2Lattice(configuration['lattice'])
    dual_lattice = lattice.plaquette_dual()

    if isinstance(result_index, int):
        result_index = [result_index]

    if len(result_index) > 1:
        futures = {}
        with ProcessPoolExecutor() as executor:
            for idx in result_index:
                counts = job_result[idx].data.c.get_counts()
                futures[idx] = executor.submit(preprocess, counts, dual_lattice)
        preprocessed = {idx: future.result() for idx, future in futures.items()}
    else:
        idx = result_index[0]
        counts = job_result[idx].data.c.get_counts()
        preprocessed = {idx: preprocess(counts, dual_lattice, batch_size=4000)}

    LOG.info('State conversion took %.2f seconds.', time.time() - start)

    file_mode = 'w' if out_filename else 'r+'
    out_filename = out_filename or filename

    for idx, (vtx_data, plaq_data) in preprocessed.items():
        if idx < configuration['num_steps']:
            dataset = 'exp'
            istep = idx
        else:
            dataset = 'ref'
            istep = idx - configuration['num_steps']

        with h5py.File(out_filename, file_mode) as out:
            try:
                del out[f'{dataset}_step{istep}']
            except KeyError:
                pass

            group = out.create_group(f'{dataset}_step{istep}')
            group.create_dataset('num_vtx', data=lattice.num_vertices)
            group.create_dataset('num_plaq', data=lattice.num_plaquettes)
            group.create_dataset('vtx_data', data=np.packbits(vtx_data, axis=1))
            group.create_dataset('plaq_data', data=np.packbits(plaq_data, axis=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--result-index', nargs='+', type=int)
    parser.add_argument('--out-filename')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    main(options.filename, options.result_index, out_filename=options.out_filename)
