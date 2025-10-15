"""SKQD with random bit flips."""
import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.sqd import sqd
from skqd_z2lgt.crbm import ConditionalRBM

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def generate_states(model, vtx_data, plaq_data, num_gen, batch_size):
    shots = plaq_data.shape[0]
    num_p = plaq_data.shape[1]
    data = np.empty((shots, num_gen, num_p), dtype=np.uint8)
    with jax.default_device(model.weights_vu.value.device):
        start = 0
        while start < vtx_data.shape[0]:
            end = start + batch_size
            if (residual := end - vtx_data.shape[0]) <= 0:
                batch = jnp.array(vtx_data[start:end])
            else:
                padding = np.zeros((residual, vtx_data.shape[1]), dtype=vtx_data.dtype)
                batch = jnp.concatenate([vtx_data[start:], padding], axis=0)
            sample = model.sample(batch, size=num_gen)
            if residual > 0:
                sample = sample[:-residual]
            flips = sample.transpose((1, 0, 2))
            data[start:end] = plaq_data[start:end, None, :] ^ flips
            start = end
    return data.reshape((-1, num_p))[:, ::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--gpu', nargs='+')
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--gen-batch-size', type=int, default=10_000)
    parser.add_argument('--niter', type=int, default=1)
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(options.gpu)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    with h5py.File(options.filename, 'r') as source:
        configuration = {}
        for key in source['configuration'].keys():
            record = source[f'configuration/{key}'][()]
            if isinstance(record, bytes):
                record = record.decode()
            configuration[key] = record

        num_plaq = source['data/num_plaq'][()]
        num_vtx = source['data/num_vtx'][()]
        exp_plaq_data = np.unpackbits(source['data/exp_plaq_data'][()], axis=2)[..., :num_plaq]
        exp_vtx_data = np.unpackbits(source['data/exp_vtx_data'][()], axis=2)[..., :num_vtx]

        raw_states = np.unpackbits(source['skqd_raw/sqd_states'][()], axis=1)[:, :num_plaq]
        raw_eigvec = source['skqd_raw/eigvec'][()]

        models = []
        for istep in range(configuration['num_steps']):
            if options.gpu and len(options.gpu) > 1:
                device = jax.devices()[istep % len(options.gpu)]
            else:
                device = None

            with jax.default_device(device):
                models.append(ConditionalRBM.load(source[f'crbm_step{istep}']))
                # Compile the model
                models[-1].sample(jnp.zeros((options.gen_batch_size, num_vtx), dtype=np.uint8),
                                  size=options.num)

    lattice = TriangularZ2Lattice(configuration['lattice'])
    dual_lattice = lattice.plaquette_dual()
    hamiltonian = dual_lattice.make_hamiltonian(configuration['plaquette_energy'])

    relevant_states = raw_states[np.square(np.abs(raw_eigvec)) > 1.e-20]
    exp_data_size = configuration['num_steps'] * configuration['shots']
    gen_data_size = exp_data_size * options.num
    max_size = gen_data_size + min(exp_data_size, 2 * relevant_states.shape[0])

    energies = []
    subspace_dims = []

    if not options.gpu or len(options.gpu) == 1:
        device_id = 0
    else:
        device_id = -1

    for it in range(options.niter):
        LOG.info('Iteration %d: %d relevant + %d generated states', it, relevant_states.shape[0],
                 exp_vtx_data.shape[0] * exp_vtx_data.shape[1] * options.num)

        is_last = it == options.niter - 1

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_states,
                                models[istep], exp_vtx_data[istep], exp_plaq_data[istep],
                                options.num, options.gen_batch_size)
                for istep in range(configuration['num_steps'])
            ]
        gen_states = [future.result() for future in futures]
        states = np.concatenate([relevant_states] + gen_states, axis=0)
        if (excess := states.shape[0] - max_size) > 0:
            max_size += 2 * excess
        sqd_result = sqd(hamiltonian, states, jax_device_id=device_id, states_size=max_size,
                         return_hproj=is_last)
        energy, eigvec, sqd_states = sqd_result[:3]
        energies.append(energy)
        subspace_dims.append(sqd_states.shape[0])

        if not is_last:
            relevant_states = sqd_states[np.square(np.abs(eigvec)) > 1.e-20]

    ham_proj = sqd_result[-1]

    groupname = 'skqd_rcv'  # pylint: disable=invalid-name
    with h5py.File(options.filename, 'r+') as out:
        try:
            del out[groupname]
        except KeyError:
            pass

        group = out.create_group(groupname)
        group.create_dataset('num_plaq', data=num_plaq)
        group.create_dataset('sqd_states', data=np.packbits(sqd_states, axis=1))
        group.create_dataset('energy', data=energy)
        group.create_dataset('eigvec', data=eigvec)
        group.create_dataset('energies', data=energies)
        group.create_dataset('subspace_dims', data=subspace_dims)
        subgroup = group.create_group('ham_proj')
        subgroup.create_dataset('data', data=ham_proj.data)
        subgroup.create_dataset('indices', data=ham_proj.indices)
        subgroup.create_dataset('indptr', data=ham_proj.indptr)
