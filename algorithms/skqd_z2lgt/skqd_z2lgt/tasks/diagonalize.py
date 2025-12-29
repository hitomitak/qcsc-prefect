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
from skqd_z2lgt.sqd import sqd, make_hproj
from skqd_z2lgt.extensions import extensions
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.utils import read_bits
from skqd_z2lgt.tasks.preprocess import RecoData, load_reco
from skqd_z2lgt.tasks.train_generator import load_model
from skqd_z2lgt.tasks.common import make_dual_lattice


def _generate_states(model, reco_data, generate_fn, batch_size):
    vtx_data, plaq_data = reco_data
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

    return np.array(gen_data.reshape((-1, num_p)))


def _make_batch_generator(num_gen):
    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    @nnx.jit
    def generate_fn(model, vtx_batch, plaq_batch):
        sample = model.sample(vtx_batch, size=num_gen)
        flips = sample.transpose((1, 0, 2))
        return model, plaq_batch[:, None, :] ^ flips

    return generate_fn


def _generate_cr(
    parameters: Parameters,
    exp_data: list[list[RecoData]],
    crbm_models: list[list[ConditionalRBM]],
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
            executor.submit(_generate_states,
                            crbm_models[idt][ikr], exp_data[idt][ikr], generate_fn,
                            parameters.crbm.gen_batch_size)
            for idt in range(len(parameters.skqd.time_steps))
            for ikr in range(len(parameters.skqd.num_krylov))
        ]
    gen_states = [future.result() for future in futures]
    logger.info('Generation took %.2f seconds.', time.time() - start)
    return gen_states


def _generate_random(
    parameters: Parameters,
    exp_data: list[list[RecoData]],
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


def _get_relevant_states(states: np.ndarray, eigvec: np.ndarray, cutoff: float) -> np.ndarray:
    return states[np.square(np.abs(eigvec)) > cutoff]


def check_saved_result(
    parameters: Parameters,
    result_name: str
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Check and return the saved SKQD result."""
    path = Path(parameters.pkgpath) / f'{result_name}.h5'
    if os.path.exists(path):
        with h5py.File(path, 'r', libver='latest') as source:
            sqd_states = read_bits(source['sqd_states'])
            energy = source['energy'][()]
            eigvec = source['eigvec'][()]
        return sqd_states, energy, eigvec
    return None


def load_projected_hamiltonian(source: h5py.Group):
    """Construct a CSR array from saved Hamiltonian data."""
    subgroup = source['ham_proj']
    data = subgroup['data'][()]
    indices = subgroup['indices'][()]
    indptr = subgroup['indptr'][()]
    shape = (indptr.shape[0] - 1,) * 2
    return csr_array((data, indices, indptr), shape=shape)


def save_skqd_result(out, sqd_states, energy, eigvec, ham_proj=None):
    """Save the SKQD result to file."""
    dataset = out.create_dataset('sqd_states', data=np.packbits(sqd_states, axis=1))
    dataset.attrs['num_bits'] = sqd_states.shape[-1]
    out.create_dataset('energy', data=energy)
    out.create_dataset('eigvec', data=eigvec)
    if ham_proj is not None:
        group = out.create_group('ham_proj')
        group.create_dataset('data', data=ham_proj.data)
        group.create_dataset('indices', data=ham_proj.indices)
        group.create_dataset('indptr', data=ham_proj.indptr)


def diagonalize_init(
    parameters: Parameters,
    jax_device_id: int = 0,
    logger: Optional[logging.Logger] = None
) -> tuple[float, np.ndarray]:
    """Project and diagonalize the Hamiltonian directly with MWPM-corrected plaquette states."""
    logger = logger or logging.getLogger(__name__)

    saved_result = check_saved_result(parameters, 'skqd_init')
    if saved_result:
        logger.info('There is already an SKQD result saved in the file.')
        states, energy, eigvec = saved_result[:3]
        relevant_states = _get_relevant_states(states, eigvec, parameters.skqd.probability_cutoff)
        return energy, relevant_states

    logger.info('Performing SQD with observed (charge-corrected) plaquette states')

    exp_data = load_reco(parameters, etype='exp')
    dual_lattice = make_dual_lattice(parameters)
    hamiltonian = dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)
    plaq_data = []
    for dt_data in exp_data:
        for _, pdata in dt_data:
            plaq_data.append(pdata)
    states = np.concatenate(plaq_data, axis=0)
    logger.info('Number of bitstrings from circuit sampling: %d', states.shape[0])
    for fname in parameters.skqd.extensions:
        states = extensions[fname](states, dual_lattice)
        logger.info('Number of bitstrings after applying %s: %d', fname, states.shape[0])
    energy, eigvec, states, ham_proj = sqd(hamiltonian, states, jax_device_id=jax_device_id)
    path = Path(parameters.pkgpath) / 'skqd_init.h5'
    with h5py.File(path, 'w', libver='latest') as out:
        save_skqd_result(out, states, energy, eigvec, ham_proj)

    relevant_states = _get_relevant_states(states, eigvec, parameters.skqd.probability_cutoff)

    return energy, relevant_states


def diagonalize(
    parameters: Parameters,
    gen_mode: str,  # 'cr' or 'rn'
    jax_device_id: int = 0,
    logger: Optional[logging.Logger] = None
) -> float:
    """Project and diagonalize the Hamiltonian with configuration recovery or random bitflips."""
    logger = logger or logging.getLogger(__name__)

    group_name = f'skqd_{gen_mode}'
    saved_result = check_saved_result(parameters, group_name)
    if saved_result:
        logger.info('There is already an SKQD result saved in the file.')
        return saved_result[1]

    exp_data = load_reco(parameters, etype='exp')
    dual_lattice = make_dual_lattice(parameters)
    hamiltonian = dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)

    if gen_mode == 'cr':
        generate_fn = _make_batch_generator(parameters.skqd.num_gen)
        logger.info('Loading and compiling CRBM models')
        idev = 0
        comp_shape = (parameters.crbm.gen_batch_size, parameters.skqd.num_gen)
        crbm_models = []
        for idt in range(len(parameters.skqd.time_steps)):
            crbm_models.append([])
            for ikrylov in range(parameters.skqd.num_krylov):
                crbm_models[-1].append(
                    load_model(parameters, idt, ikrylov, compile_for=comp_shape, jax_device_id=idev)
                )
                idev += 1
    else:
        ref_data = load_reco(parameters, etype='ref')
        logger.info('Using the mean of reference circuit data as single-plaquette probability')
        mean_activation = [
            [np.mean(plaq_data, axis=0) for _, plaq_data in dt_data]
            for dt_data in ref_data
        ]

    energies = []
    subspace_dims = []
    relevant_states = None
    max_size = None
    is_last = False
    for it in range(parameters.skqd.max_iterations):
        logger.info('SKQD iteration %d', it)
        is_last = it == parameters.skqd.max_iterations - 1

        if gen_mode == 'cr':
            gen_states = _generate_cr(parameters, exp_data, crbm_models, generate_fn, logger)
        else:
            gen_states = _generate_random(parameters, exp_data, mean_activation, 12345 + it, logger)

        states = np.concatenate([relevant_states] + gen_states, axis=0)
        for fname in parameters.skqd.extensions:
            states = extensions[fname](states, dual_lattice)
        if max_size is None:
            max_size = states.shape[0] + relevant_states.shape[0]
            logger.info('Set maximum array size to %d', max_size)
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
            prev_energy - energy < parameters.skqd.delta_e
            or
            relevant_states.shape[0] >= parameters.skqd.max_subspace_dim
        )

        if not is_last and terminate:
            states_p = np.packbits(states, axis=1)
            sqd_result += (make_hproj(hamiltonian, states_p),)
            is_last = True

        if is_last:
            break

        prev_energy = energy
        relevant_states = _get_relevant_states(sqd_states, eigvec,
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

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--gpus', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode', default='rcv')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()),
                        stream=sys.stdout)
    logging.getLogger('jax').setLevel(logging.WARNING)

    if options.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    with open(Path(options.pkgpath) / 'parameters.json', 'r', encoding='utf-8') as src:
        params = Parameters.model_validate_json(src.read())

    en_init, st_init = diagonalize_init(params)

    if options.mode == 'init':
        sys.exit(0)
    if options.mode not in ['rcv', 'rnd']:
        raise ValueError(f'Invalid mode {options.mode}')

    diagonalize(params, en_init, st_init, options.mode)
