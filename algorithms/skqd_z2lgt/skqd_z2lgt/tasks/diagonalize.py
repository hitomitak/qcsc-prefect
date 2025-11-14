# pylint: disable=no-member
"""SKQD with configuration recovery."""
from collections.abc import Callable
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
from scipy.sparse import csr_array
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from qiskit.quantum_info import SparsePauliOp
from heavyhex_qft.triangular_z2 import TriangularZ2Lattice
from skqd_z2lgt.sqd import sqd, to_bcoo, bcoo_to_csr
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.parameters import Parameters


def check_saved_result(
    parameters: Parameters,
    group_name: str
) -> tuple[float, np.ndarray] | None:
    with h5py.File(parameters.output_filename, 'r', libver='latest') as source:
        if (group := source.get(group_name)) is None:
            return None
        return group['energy'][()], group['eigvec'][()]


def load_init(
    parameters: Parameters
):
    with h5py.File(parameters.output_filename, 'r', libver='latest') as source:
        if (group := source.get('skqd_init')) is None:
            return None
        sqd_result = load_skqd_result(group)
        return sqd_result[:3]


def diagonalize_init(
    parameters: Parameters,
    exp_data: list[tuple[np.ndarray, np.ndarray]],
    hamiltonian: SparsePauliOp,
    logger: Optional[logging.Logger] = None
):
    logger = logger or logging.getLogger(__name__)
    logger.info('Performing SQD with observed (charge-corrected) plaquette states')
    states = np.concatenate([pdata for _, pdata in exp_data], axis=0)[:, ::-1]
    energy, eigvec, states, ham_proj = sqd(hamiltonian, states)
    with h5py.File(parameters.output_filename, 'r+', libver='latest') as out:
        save_skqd_result(out, 'skqd_init', states, energy, eigvec, ham_proj)

    return states, energy, eigvec


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


def generate_with_crbm(
    parameters: Parameters,
    exp_data: list[tuple[np.ndarray, np.ndarray]],
    crbm_models: list[ConditionalRBM],
    generate_fn: Callable,
    logger: Optional[logging.Logger] = None
) -> list[np.ndarray]:
    logger = logger or logging.getLogger(__name__)
    num_steps = len(exp_data)
    shots = exp_data[0][0].shape[0]

    logger.info('Generating for %d steps, %d shots, %d samples per shot',
                num_steps, shots, parameters.skqd.num_gen)
    start = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_states,
                            crbm_models[istep], exp_data[istep][0], exp_data[istep][1],
                            generate_fn, parameters.crbm.gen_batch_size)
            for istep in range(len(exp_data))
        ]
    gen_states = [future.result() for future in futures]
    logger.info('Generation took %.2f seconds.', time.time() - start)
    return gen_states


def generate_random(
    parameters: Parameters,
    exp_data: list[tuple[np.ndarray, np.ndarray]],
    mean_activation: list[np.ndarray],
    seed: int,
    logger: Optional[logging.Logger] = None
) -> list[np.ndarray]:
    logger = logger or logging.getLogger(__name__)
    num_steps = len(exp_data)
    shots = exp_data[0][0].shape[0]
    num_plaq = exp_data[0][1].shape[1]

    logger.info('Generating random bitflips for %d steps, %d shots, %d samples per shot',
                num_steps, shots, parameters.skqd.num_gen)
    start = time.time()
    rng = np.random.default_rng(seed)
    uniform = rng.random((num_steps, shots, parameters.skqd.num_gen, num_plaq))
    flips = [np.asarray(uniform[istep] < act[None, None, :], dtype=np.uint8)
             for istep, act in enumerate(mean_activation)]
    gen_states = [(plaq_data[:, None, :] ^ fl).reshape((-1, num_plaq))
                  for (_, plaq_data), fl in zip(exp_data, flips)]
    logger.info('Generation took %.2f seconds.', time.time() - start)
    return gen_states


def load_skqd_result(group):
    dataset = group['sqd_states']
    sqd_states = np.unpackbits(dataset[()], axis=1)[:, :dataset.attrs['num_bits']]
    energy = group['energy'][()]
    eigvec = group['eigvec'][()]
    if (subgroup := group.get('ham_proj')) is None:
        ham_proj = None
    else:
        data = subgroup['data'][()]
        indices = subgroup['indices'][()]
        indptr = subgroup['indptr'][()]
        shape = (indptr.shape[0] - 1,) * 2
        ham_proj = csr_array((data, indices, indptr), shape=shape)
    return sqd_states, energy, eigvec, ham_proj


def save_skqd_result(out, group_name, sqd_states, energy, eigvec, ham_proj=None):
    try:
        del out[group_name]
    except KeyError:
        pass

    group = out.create_group(group_name)
    dataset = group.create_dataset('sqd_states', data=np.packbits(sqd_states, axis=1))
    dataset.attrs['num_bits'] = sqd_states.shape[-1]
    group.create_dataset('energy', data=energy)
    group.create_dataset('eigvec', data=eigvec)
    if ham_proj is not None:
        subgroup = group.create_group('ham_proj')
        subgroup.create_dataset('data', data=ham_proj.data)
        subgroup.create_dataset('indices', data=ham_proj.indices)
        subgroup.create_dataset('indptr', data=ham_proj.indptr)
    return group


def diagonalize(
    parameters: Parameters,
    reco_data: list[tuple[np.ndarray, np.ndarray]],
    crbm_models: list[ConditionalRBM] | None,
    logger: Optional[logging.Logger] = None
) -> tuple[float, np.ndarray]:
    logger = logger or logging.getLogger(__name__)

    if crbm_models:
        group_name = 'skqd_rcv'
    else:
        group_name = 'skqd_rnd'

    saved_result = check_saved_result(parameters, group_name)
    if saved_result:
        logger.info('There is already an SKQD result saved in the file.')
        return saved_result

    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    dual_lattice = lattice.plaquette_dual()
    hamiltonian = dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)

    exp_data = reco_data[0]

    init = load_init(parameters)
    if init is None:
        init_states, init_energy, init_eigvec = diagonalize_init(parameters, exp_data, hamiltonian,
                                                                 logger)
    else:
        init_states, init_energy, init_eigvec = init

    if parameters.skqd.max_iterations == 0:
        return init_energy, init_eigvec

    num_steps = parameters.skqd.n_trotter_steps
    shots = parameters.runtime.shots

    relevant_states = init_states[np.square(np.abs(init_eigvec)) > 1.e-20]
    exp_data_size = num_steps * shots
    gen_data_size = exp_data_size * parameters.skqd.num_gen
    max_size = gen_data_size + min(exp_data_size, 10 * relevant_states.shape[0])
    logger.info('Set maximum array size to %d', max_size)

    if crbm_models:
        generate_fn = make_batch_generator(parameters.skqd.num_gen)
    else:
        ref_data = reco_data[1]
        mean_activation = [np.mean(plaq_data, axis=0) for _, plaq_data in ref_data]

    energies = []
    subspace_dims = []
    prev_energy = None

    is_last = False
    for it in range(parameters.skqd.max_iterations):
        logger.info('Iteration %d: %d relevant states', it, relevant_states.shape[0])
        is_last = it == parameters.skqd.max_iterations - 1

        if crbm_models:
            gen_states = generate_with_crbm(parameters, exp_data, crbm_models, generate_fn, logger)
        else:
            gen_states = generate_random(parameters, exp_data, mean_activation, 12345 + it, logger)

        states = np.concatenate([relevant_states] + gen_states, axis=0)
        if (excess := states.shape[0] - max_size) > 0:
            max_size += 10 * excess
            logger.info('Updated maximum array size to %d', max_size)
        logger.info('Diagonalizing the Hamiltonian projected onto %d states..', states.shape[0])
        start = time.time()
        sqd_result = sqd(hamiltonian, states, states_size=max_size, return_hproj=is_last)
        energy, eigvec, sqd_states = sqd_result[:3]
        energies.append(energy)
        subspace_dims.append(sqd_states.shape[0])
        logger.info('Projection and diagonalization took %.2f seconds. Current energy %.4f',
                    time.time() - start, energy)

        terminate = (
            (prev_energy is not None and prev_energy - energy < parameters.skqd.delta_e)
            or
            relevant_states.shape[0] >= parameters.skqd.max_subspace_dim
        )

        if not is_last and terminate:
            hproj = to_bcoo(hamiltonian, np.packbits(sqd_states, axis=1))
            sqd_result += (bcoo_to_csr(hproj),)
            is_last = True

        if is_last:
            break

        relevant_states = sqd_states[np.square(np.abs(eigvec)) > 1.e-20]

    ham_proj = sqd_result[-1]

    with h5py.File(parameters.output_filename, 'r+', libver='latest') as out:
        group = save_skqd_result(out, group_name, sqd_states, energy, eigvec, ham_proj)
        group.create_dataset('energies', data=energies)
        group.create_dataset('subspace_dims', data=subspace_dims)

    return energy, eigvec


if __name__ == '__main__':
    import os
    import argparse
    from skqd_z2lgt.tasks.preprocess import load_reco
    from skqd_z2lgt.tasks.train_generator import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--gpu', nargs='+')
    parser.add_argument('--num-gen', type=int, default=5)
    parser.add_argument('--gen-batch-size', type=int, default=10_000)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--niter', type=int, default=1)
    parser.add_argument('--terminate-deltae', type=float, default=0.005)
    parser.add_argument('--terminate-ndim', type=int, default=1_000_000)
    options = parser.parse_args()

    LOG = logging.getLogger(__name__)

    if options.gpu and options.gpu[0] != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(options.gpu)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    params = Parameters()
    params.output_filename = options.filename
    with h5py.File(options.filename, 'r', libver='latest') as src:
        params.lgt.lattice = src.attrs['lattice']
        params.lgt.plaquette_energy = src.attrs['plaquette_energy']
        params.lgt.charged_vertices = src.attrs['charged_vertices']
        params.skqd.n_trotter_steps = src.attrs['num_steps']
        params.runtime.shots = src.attrs['shots']
    params.skqd.num_gen = options.num_gen
    params.skqd.max_iterations = options.niter
    params.crbm.gen_batch_size = options.gen_batch_size
    params.skqd.delta_e = options.terminate_deltae
    params.skqd.max_subspace_dim = options.terminate_ndim

    rdata = load_reco(params)
    if options.random:
        models = None
    else:
        models = [load_model(istep, params.output_filename, istep % jax.device_count())
                  for istep in range(params.skqd.n_trotter_steps)]

        for istep, crbm in enumerate(models):
            LOG.info('Compiling CRBM for Trotter step %d', istep)
            with jax.default_device(crbm.weights_vu.value.device):
                crbm.sample(
                    jnp.zeros((params.crbm.gen_batch_size, crbm.weights_vu.shape[1]),
                              dtype=np.uint8),
                    size=params.skqd.num_gen
                )

    diagonalize(params, rdata, models)
