"""SKQD with random bit flips."""
import os
import argparse
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.sqd import sqd, to_bcoo, bcoo_to_csr
from skqd_z2lgt.crbm import ConditionalRBM

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def generate_states(model, vtx_data, plaq_data, generate_fn, batch_size):
    shots = plaq_data.shape[0]
    num_p = plaq_data.shape[1]

    with jax.default_device(model.weights_vu.value.device):
        num_batches = int(np.ceil(shots / batch_size).astype(int))
        residue = num_batches * batch_size - shots
        if residue != 0:
            padding = np.zeros((residue,) + vtx_data.shape[1:], dtype=np.uint8)
            vtx_data = jnp.concatenate([vtx_data, padding], axis=0)
            padding = np.zeros((residue,) + plaq_data.shape[1:], dtype=np.uint8)
            plaq_data = jnp.concatenate([plaq_data, padding], axis=0)
        else:
            vtx_data = jax.device_put(vtx_data)
            plaq_data = jax.device_put(plaq_data)

        vtx_data = vtx_data.reshape((num_batches, batch_size,) + vtx_data.shape[1:])
        plaq_data = plaq_data.reshape((num_batches, batch_size,) + plaq_data.shape[1:])

        gen_data = generate_fn(model, vtx_data, plaq_data)[1]

    return np.array(gen_data.reshape((-1, num_p))[:, ::-1])


def make_batch_generator(num_gen):
    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    @nnx.jit
    def generate_fn(model, vtx_batch, plaq_batch):
        sample = model.sample(vtx_batch, size=num_gen)
        flips = sample.transpose((1, 0, 2))
        return model, plaq_batch[:, None, :] ^ flips

    return generate_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--gpu', nargs='+')
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--gen-batch-size', type=int, default=10_000)
    parser.add_argument('--niter', type=int, default=1)
    parser.add_argument('--terminate', nargs='+')
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
    max_size = gen_data_size + min(exp_data_size, 10 * relevant_states.shape[0])
    LOG.info('Set maximum array size to %d', max_size)

    generate_fn = make_batch_generator(options.num)

    energies = []
    subspace_dims = []

    if not options.gpu or len(options.gpu) == 1:
        device_id = 0
    else:
        device_id = -1

    terminate_conditions = {}
    if options.terminate:
        for term in options.terminate:
            key, _, value = term.partition('=')
            terminate_conditions[key] = float(value)
    prev_energy = None

    is_last = False
    for it in range(options.niter):
        LOG.info('Iteration %d: %d relevant states', it, relevant_states.shape[0])
        is_last = it == options.niter - 1

        LOG.info('Generating for %d steps, %d shots, %d samples per shot',
                 exp_vtx_data.shape[0], exp_vtx_data.shape[1], options.num)
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_states,
                                models[istep], exp_vtx_data[istep], exp_plaq_data[istep],
                                generate_fn, options.gen_batch_size)
                for istep in range(configuration['num_steps'])
            ]
        gen_states = [future.result() for future in futures]
        LOG.info('Generation took %.2f seconds.', time.time() - start)

        states = np.concatenate([relevant_states] + gen_states, axis=0)
        if (excess := states.shape[0] - max_size) > 0:
            max_size += 10 * excess
            LOG.info('Updated maximum array size to %d', max_size)
        LOG.info('Diagonalizing the Hamiltonian projected onto %d states..', states.shape[0])
        start = time.time()
        sqd_result = sqd(hamiltonian, states, jax_device_id=device_id, states_size=max_size,
                         return_hproj=is_last)
        energy, eigvec, sqd_states = sqd_result[:3]
        energies.append(energy)
        subspace_dims.append(sqd_states.shape[0])
        LOG.info('Projection and diagonalization took %.2f seconds. Current energy %.4f',
                 time.time() - start, energy)

        terminate = False
        for key, value in terminate_conditions.items():
            if key == 'diff':
                terminate = prev_energy is not None and prev_energy - energy < value
            elif key == 'dim':
                terminate = relevant_states.shape[0] >= value

        if not is_last and terminate:
            subspace_dim = sqd_states.shape[0]
            hproj = to_bcoo(hamiltonian, np.packbits(sqd_states, axis=1), sharded=device_id < 0)
            sqd_result += (bcoo_to_csr(hproj),)
            is_last = True

        if is_last:
            break

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
