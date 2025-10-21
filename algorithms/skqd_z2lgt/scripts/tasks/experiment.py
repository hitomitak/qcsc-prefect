"""Run the quantum experiments."""
import argparse
import logging
import time
from typing import Optional
import h5py
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.circuits import make_step_circuits, compose_trotter_circuits

LOG = logging.getLogger(__name__)


def main(filename: str, sampler_options: Optional[dict] = None, job_id: Optional[str] = None):
    with h5py.File(filename, 'r', swmr=True) as source:
        configuration = dict(source.attrs)

    service = QiskitRuntimeService(instance=configuration['instance'])
    backend = service.backend(configuration['backend'])

    lattice = TriangularZ2Lattice(configuration['lattice'])

    layout = lattice.layout_heavy_hex(backend.coupling_map,
                                      backend_properties=backend.properties(),
                                      basis_2q=configuration['basis_2q'])

    if job_id:
        LOG.info('Retrieving job %s', job_id)
        job = service.job(job_id)
    else:
        LOG.info('Transpiling single-step circuits..')
        start = time.time()
        full_step, fwd_step, bkd_step, measure = transpile(
            make_step_circuits(lattice, configuration['plaquette_energy'],
                               configuration['delta_t'], configuration['basis_2q']),
            backend=backend, initial_layout=layout, optimization_level=2
        )
        id_step = fwd_step.compose(bkd_step)
        exp_circuits = compose_trotter_circuits(full_step, measure, configuration['num_steps'])
        ref_circuits = compose_trotter_circuits(id_step, measure, configuration['num_steps'])
        LOG.info('Transpilation and composition of the circuits took %.2f seconds.',
                 time.time() - start)

        sampler = Sampler(backend, options=sampler_options)
        job = sampler.run(exp_circuits + ref_circuits)
        LOG.info('Submitted job %s to %s.', job.job_id(), configuration['backend'])

    result = job.result()

    with h5py.File(filename, 'r+') as out:
        try:
            del out['data/raw']
        except KeyError:
            pass
        group = out.create_group('data/raw')
        group.attrs['job_id'] = job.job_id()
        group.attrs['layout'] = layout
        for ires, res in enumerate(result):
            if ires < configuration['num_steps']:
                etype = 'exp'
            else:
                etype = 'ref'
            istep = ires % configuration['num_steps']
            # Note: Qiskit BitArray is packed in right-aligned big endian
            dataset = group.create_dataset(f'{etype}_step{istep}', data=res.data.c.array)
            dataset.attrs['num_bits'] = res.data.c.num_bits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--shots')
    parser.add_argument('--job-id')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))

    primitive_options = {}
    if options.shots:
        primitive_options['default_shots'] = options.shots
    main(options.filename, sampler_options=primitive_options, job_id=options.job_id)
