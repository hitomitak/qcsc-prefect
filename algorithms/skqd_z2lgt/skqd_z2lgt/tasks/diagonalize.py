# pylint: disable=no-member
"""SKQD with configuration recovery."""
import os
from collections.abc import Callable
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
from skqd_z2lgt.mwpm import minimum_weight_link_state
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.utils import read_bits


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


def get_relevant_states(states: np.ndarray, eigvec: np.ndarray, cutoff: float) -> np.ndarray:
    return states[np.square(np.abs(eigvec)) > cutoff]


def check_saved_result(
    parameters: Parameters,
    result_name: str
) -> tuple[np.ndarray, float, np.ndarray] | None:
    path = Path(parameters.pkgpath) / f'{result_name}.h5'
    if os.path.exists(path):
        with h5py.File(path, 'r', libver='latest') as source:
            states, energy, eigvec, _ = load_skqd_result(source)
            return states, energy, eigvec
    return None


def load_skqd_result(group):
    sqd_states = read_bits(group['sqd_states'])
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


def save_skqd_result(out, sqd_states, energy, eigvec, ham_proj=None):
    dataset = out.create_dataset('sqd_states', data=np.packbits(sqd_states, axis=1))
    dataset.attrs['num_bits'] = sqd_states.shape[-1]
    out.create_dataset('energy', data=energy)
    out.create_dataset('eigvec', data=eigvec)
    if ham_proj is not None:
        group = out.create_group('ham_proj')
        group.create_dataset('data', data=ham_proj.data)
        group.create_dataset('indices', data=ham_proj.indices)
        group.create_dataset('indptr', data=ham_proj.indptr)


def make_hamiltonian(parameters: Parameters) -> SparsePauliOp:
    """Return the Ising hamiltonian for the given charge sector."""
    lattice = TriangularZ2Lattice(parameters.lgt.lattice)
    base_link_state = minimum_weight_link_state(parameters.lgt.charged_vertices, lattice)
    dual_lattice = lattice.plaquette_dual(base_link_state)
    return dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)


def diagonalize_init(
    parameters: Parameters,
    exp_data: list[tuple[np.ndarray, np.ndarray]],
    jax_device_id: int = 0,
    logger: Optional[logging.Logger] = None
) -> tuple[float, np.ndarray]:
    logger = logger or logging.getLogger(__name__)

    saved_result = check_saved_result(parameters, 'skqd_init')
    if saved_result:
        logger.info('There is already an SKQD result saved in the file.')
        states, energy, eigvec = saved_result[:3]
        relevant_states = get_relevant_states(states, eigvec, parameters.skqd.probability_cutoff)
        return energy, relevant_states

    logger.info('Performing SQD with observed (charge-corrected) plaquette states')

    hamiltonian = make_hamiltonian(parameters)
    states = np.concatenate([pdata for _, pdata in exp_data], axis=0)[:, ::-1]
    energy, eigvec, states, ham_proj = sqd(hamiltonian, states, jax_device_id=jax_device_id)
    path = Path(parameters.pkgpath) / 'skqd_init.h5'
    with h5py.File(path, 'w', libver='latest') as out:
        save_skqd_result(out, states, energy, eigvec, ham_proj)

    relevant_states = get_relevant_states(states, eigvec, parameters.skqd.probability_cutoff)

    return energy, relevant_states


def diagonalize(
    parameters: Parameters,
    exp_data: list[tuple[np.ndarray, np.ndarray]],
    states_init: np.ndarray,
    crbm_models: Optional[list[ConditionalRBM]] = None,
    ref_data: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    jax_device_id: int = 0,
    logger: Optional[logging.Logger] = None
) -> float:
    logger = logger or logging.getLogger(__name__)

    if crbm_models:
        group_name = 'skqd_rcv'
    else:
        group_name = 'skqd_rnd'

    saved_result = check_saved_result(parameters, group_name)
    if saved_result:
        logger.info('There is already an SKQD result saved in the file.')
        return saved_result[1]

    hamiltonian = make_hamiltonian(parameters)
    num_steps = parameters.skqd.n_trotter_steps
    shots = parameters.runtime.shots

    exp_data_size = num_steps * shots
    gen_data_size = exp_data_size * parameters.skqd.num_gen
    max_size = gen_data_size + min(exp_data_size, 10 * states_init.shape[0])
    logger.info('Set maximum array size to %d', max_size)

    if crbm_models:
        generate_fn = make_batch_generator(parameters.skqd.num_gen)
    else:
        if ref_data is None:
            logger.warning('Assuming single-plaquette flip probability of 0.5')
            mean_activation = [np.full(exp_data[0][1].shape[1], 0.5)] * num_steps
        else:
            logger.info('Using the mean of reference circuit data as single-plaquette probability')
            mean_activation = [np.mean(plaq_data, axis=0) for _, plaq_data in ref_data]

    energies = []
    subspace_dims = []
    relevant_states = states_init
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
        sqd_result = sqd(hamiltonian, states, states_size=max_size, return_hproj=is_last,
                         jax_device_id=jax_device_id)
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

        relevant_states = get_relevant_states(sqd_states, eigvec,
                                              parameters.skqd.probability_cutoff)

    ham_proj = sqd_result[-1]

    path = Path(parameters.pkgpath) / f'{group_name}.h5'
    with h5py.File(path, 'w', libver='latest') as out:
        save_skqd_result(out, sqd_states, energy, eigvec, ham_proj)
        out.create_dataset('energies', data=energies)
        out.create_dataset('subspace_dims', data=subspace_dims)

    return energy


if __name__ == '__main__':
    import sys
    import argparse
    from skqd_z2lgt.tasks.preprocess import load_reco
    from skqd_z2lgt.tasks.train_generator import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--gpus', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode', default='full')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))
    LOG = logging.getLogger(__name__)

    if options.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    with open(Path(options.pkgpath) / 'parameters.json', 'r', encoding='utf-8') as src:
        params = Parameters.model_validate_json(src.read())

    edata = load_reco(params, 'exp')
    _, st_init = diagonalize_init(params, edata)

    if options.mode == 'init':
        sys.exit(0)

    if options.mode == 'random':
        models = None  # pylint: disable=invalid-name
        rdata = load_reco(params, 'ref')
    else:
        rdata = None  # pylint: disable=invalid-name
        models = [load_model(params, istep, istep % jax.device_count())[0]
                  for istep in range(params.skqd.n_trotter_steps)]

        for istep, crbm in enumerate(models):
            LOG.info('Compiling CRBM for Trotter step %d', istep)
            with jax.default_device(crbm.weights_vu.value.device):
                crbm.sample(
                    jnp.zeros((params.crbm.gen_batch_size, crbm.weights_vu.shape[1]),
                              dtype=np.uint8),
                    size=params.skqd.num_gen
                )

    diagonalize(params, edata, st_init, crbm_models=models, ref_data=rdata)
