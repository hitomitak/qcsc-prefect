"""SKQD with random bit flips."""
import os
import argparse
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
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


def main(
    filename: str,
    gen_batch_size: int = 10_000,
    num_gen: int = 5,
    niter: int = 10,
    terminate_conditions: Optional[dict[str, Any]] = None,
    multi_gpu: bool = False
):
    with h5py.File(filename, 'r') as source:
        configuration = dict(source.attrs)

        num_steps = configuration['num_steps']
        exp_plaq_data = []
        exp_vtx_data = []
        for istep in range(num_steps):
            dataset = source[f'exp_step{istep}/plaq_data']
            exp_plaq_data.append(
                np.unpackbits(dataset[()], axis=-1)[..., :dataset.attrs['num_bits']]
            )
            dataset = source[f'exp_step{istep}/vtx_data']
            exp_vtx_data.append(
                np.unpackbits(dataset[()], axis=-1)[..., :dataset.attrs['num_bits']]
            )

        shots, num_vtx = exp_vtx_data[0].shape[-2:]

        dataset = source['skqd_raw/sqd_states']
        raw_states = np.unpackbits(dataset[()], axis=-1)[:, :dataset.attrs['num_bits']]
        raw_eigvec = source['skqd_raw/eigvec'][()]

        models = []
        for istep in range(num_steps):
            if multi_gpu:
                device = jax.devices()[istep % jax.device_count()]
            else:
                device = None

            with jax.default_device(device):
                models.append(ConditionalRBM.load(source[f'crbm_step{istep}']))
                # Compile the model
                models[-1].sample(jnp.zeros((gen_batch_size, num_vtx), dtype=np.uint8),
                                  size=num_gen)

    lattice = TriangularZ2Lattice(configuration['lattice'])
    dual_lattice = lattice.plaquette_dual()
    hamiltonian = dual_lattice.make_hamiltonian(configuration['plaquette_energy'])

    relevant_states = raw_states[np.square(np.abs(raw_eigvec)) > 1.e-20]
    exp_data_size = num_steps * shots
    gen_data_size = exp_data_size * num_gen
    max_size = gen_data_size + min(exp_data_size, 10 * relevant_states.shape[0])
    LOG.info('Set maximum array size to %d', max_size)

    generate_fn = make_batch_generator(num_gen)

    energies = []
    subspace_dims = []
    terminate_conditions = terminate_conditions or {}
    prev_energy = None

    is_last = False
    for it in range(niter):
        LOG.info('Iteration %d: %d relevant states', it, relevant_states.shape[0])
        is_last = it == niter - 1

        LOG.info('Generating for %d steps, %d shots, %d samples per shot',
                 num_steps, shots, num_gen)
        start = time.time()
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_states,
                                models[istep], exp_vtx_data[istep], exp_plaq_data[istep],
                                generate_fn, gen_batch_size)
                for istep in range(num_steps)
            ]
        gen_states = [future.result() for future in futures]
        LOG.info('Generation took %.2f seconds.', time.time() - start)

        states = np.concatenate([relevant_states] + gen_states, axis=0)
        if (excess := states.shape[0] - max_size) > 0:
            max_size += 10 * excess
            LOG.info('Updated maximum array size to %d', max_size)
        LOG.info('Diagonalizing the Hamiltonian projected onto %d states..', states.shape[0])
        start = time.time()
        sqd_result = sqd(hamiltonian, states, jax_device_id=-1 if multi_gpu else None,
                         states_size=max_size, return_hproj=is_last)
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
            hproj = to_bcoo(hamiltonian, np.packbits(sqd_states, axis=1), sharded=multi_gpu)
            sqd_result += (bcoo_to_csr(hproj),)
            is_last = True

        if is_last:
            break

        relevant_states = sqd_states[np.square(np.abs(eigvec)) > 1.e-20]

    ham_proj = sqd_result[-1]

    groupname = 'skqd_rcv'  # pylint: disable=invalid-name
    with h5py.File(filename, 'r+') as out:
        try:
            del out[groupname]
        except KeyError:
            pass

        group = out.create_group(groupname)
        dataset = group.create_dataset('sqd_states', data=np.packbits(sqd_states, axis=1))
        dataset.attrs['num_bits'] = lattice.num_plaquettes
        group.create_dataset('energy', data=energy)
        group.create_dataset('eigvec', data=eigvec)
        group.create_dataset('energies', data=energies)
        group.create_dataset('subspace_dims', data=subspace_dims)
        subgroup = group.create_group('ham_proj')
        subgroup.create_dataset('data', data=ham_proj.data)
        subgroup.create_dataset('indices', data=ham_proj.indices)
        subgroup.create_dataset('indptr', data=ham_proj.indptr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--gpu', nargs='+')
    parser.add_argument('--num-gen', type=int, default=5)
    parser.add_argument('--gen-batch-size', type=int, default=10_000)
    parser.add_argument('--niter', type=int, default=1)
    parser.add_argument('--terminate', nargs='+')
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(options.gpu)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    conditions = {}
    if options.terminate:
        for term in options.terminate:
            cond = term.partition('=')
            conditions[cond[0]] = float(cond[2])

    main(options.filename, gen_batch_size=options.gen_batch_size, num_gen=options.num_gen,
         niter=options.niter, terminate_conditions=conditions,
         multi_gpu=options.gpu and len(options.gpu) > 1)
