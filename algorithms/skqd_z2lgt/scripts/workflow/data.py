"""Process the link-state bitstrings with MWPM."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py
from qiskit_ibm_runtime import QiskitRuntimeService
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.recovery_learning import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    options = parser.parse_args()

    with h5py.File(options.filename, 'r', swmr=True) as source:
        configuration = {}
        for key in source['configuration'].keys():
            record = source[f'configuration/{key}'][()]
            if isinstance(record, bytes):
                record = record.decode()
            configuration[key] = record

        job_id = source['experiment/job_id'][()].decode()  # pylint: disable=no-member

    service = QiskitRuntimeService(instance=configuration['instance'])
    try:
        job_result = service.job(job_id).result()
    except json.JSONDecodeError as exc:
        print(exc.doc)
        raise

    lattice = TriangularZ2Lattice(configuration['lattice'])
    dual_lattice = lattice.plaquette_dual()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(preprocess, res.data.c.get_counts(), dual_lattice)
                   for res in job_result]

    shape_base = (configuration['num_steps'], configuration['shots'])
    exp_vtx_data = np.empty(shape_base + (lattice.num_vertices,), dtype=np.uint8)
    exp_plaq_data = np.empty(shape_base + (lattice.num_plaquettes,), dtype=np.uint8)
    ref_vtx_data = np.empty(shape_base + (lattice.num_vertices,), dtype=np.uint8)
    ref_plaq_data = np.empty(shape_base + (lattice.num_plaquettes,), dtype=np.uint8)
    for istep, future in enumerate(futures[:configuration['num_steps']]):
        vtx_data, plaq_data = future.result()
        exp_vtx_data[istep] = vtx_data
        exp_plaq_data[istep] = plaq_data
    for istep, future in enumerate(futures[configuration['num_steps']:]):
        vtx_data, plaq_data = future.result()
        ref_vtx_data[istep] = vtx_data
        ref_plaq_data[istep] = plaq_data

    with h5py.File(options.filename, 'r+') as out:
        try:
            del out['data']
        except KeyError:
            pass

        group = out.create_group('data')
        group.create_dataset('num_vtx', data=lattice.num_vertices)
        group.create_dataset('num_plaq', data=lattice.num_plaquettes)
        group.create_dataset('exp_vtx_data', data=np.packbits(exp_vtx_data, axis=2))
        group.create_dataset('exp_plaq_data', data=np.packbits(exp_plaq_data, axis=2))
        group.create_dataset('ref_vtx_data', data=np.packbits(ref_vtx_data, axis=2))
        group.create_dataset('ref_plaq_data', data=np.packbits(ref_plaq_data, axis=2))
