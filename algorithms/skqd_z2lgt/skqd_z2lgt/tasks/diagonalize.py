# pylint: disable=no-member
"""SKQD with configuration recovery."""
import os
import time
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.sparse import csr_array
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.sqd import sqd, make_hproj, uniquify_states
from skqd_z2lgt.extensions import extensions
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.utils import read_bits, save_bits
from skqd_z2lgt.tasks.preprocess import RecoData, load_reco
from skqd_z2lgt.tasks.train_generator import load_model
from skqd_z2lgt.tasks.common import make_dual_lattice


def _prepare_data_and_models(
    parameters: Parameters,
    exp_data: list[list[RecoData]],
    logger: logging.Logger
):
    num_vertices = exp_data[0][0][0].shape[1]
    num_plaquettes = exp_data[0][0][1].shape[1]
    shots = parameters.runtime.shots_exp
    batch_size = parameters.crbm.gen_batch_size
    num_batches = int(np.ceil(shots / batch_size).astype(int))
    if (residue := num_batches * batch_size - shots) != 0:
        padding = (
            np.zeros((residue, num_vertices), dtype=np.uint8),
            np.zeros((residue, num_plaquettes), dtype=np.uint8)
        )
    else:
        padding = None

    logger.info('Loading and compiling CRBM models')
    comp_shape = (parameters.crbm.gen_batch_size, parameters.skqd.num_gen)
    crbm_models = []
    for idt in range(len(parameters.skqd.time_steps)):
        crbm_models.append([])
        for ikrylov in range(1, parameters.skqd.num_krylov + 1):
            crbm_models[-1].append(
                load_model(parameters, idt, ikrylov, compile_for=comp_shape)
            )
            vtx_data, plaq_data = exp_data[idt][ikrylov - 1]
            if padding:
                vtx_data = np.concatenate([vtx_data, padding[0]], axis=0)
                plaq_data = np.concatenate([plaq_data, padding[1]], axis=0)
            vtx_data = jax.device_put(vtx_data)
            plaq_data = jax.device_put(plaq_data)
            exp_data[idt][ikrylov - 1] = (
                vtx_data.reshape((num_batches, batch_size, num_vertices)),
                plaq_data.reshape((num_batches, batch_size, num_plaquettes))
            )

    return crbm_models


@nnx.jit(static_argnums=[3, 4])
def _generate_states(model, vtx_data, plaq_data, num_gen, shots):
    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0,))
    def generate_fn(_model, vtx_batch, plaq_batch):
        sample = _model.sample(vtx_batch, size=num_gen)
        flips = sample.transpose((1, 0, 2))
        return _model, plaq_batch[:, None, :] ^ flips

    gen_states = generate_fn(model, vtx_data, plaq_data)[1]
    return gen_states.reshape((-1, plaq_data.shape[-1]))[:shots * num_gen]


def _generate_cr(
    parameters: Parameters,
    exp_data: list[list[RecoData]],
    crbm_models: list[list[ConditionalRBM]],
    logger: logging.Logger
) -> list[np.ndarray]:
    logger.info('Generating %d samples for each of %d shots of %d circuits',
                parameters.skqd.num_gen, parameters.runtime.shots_exp,
                sum(len(data) for data in exp_data))
    start = time.time()
    gen_states = [
        _generate_states(crbm_models[idt][ikr], exp_data[idt][ikr][0], exp_data[idt][ikr][1],
                         parameters.skqd.num_gen, parameters.runtime.shots_exp)
        for idt in range(len(parameters.skqd.time_steps))
        for ikr in range(parameters.skqd.num_krylov)
    ]
    logger.info('Generation took %.2f seconds.', time.time() - start)
    return gen_states


def _prepare_mean_activation(
    parameters: Parameters,
    logger: logging.Logger
):
    logger.info('Using the mean of reference circuit data as single-plaquette probability')
    ref_data = load_reco(parameters, etype='ref')
    mean_activation = []
    for idt in range(len(parameters.skqd.time_steps)):
        mean_activation.append([])
        for ikr in range(parameters.skqd.num_krylov):
            mean_activation[-1].append(np.mean(ref_data[idt][ikr][1], axis=0))

    return mean_activation


@nnx.jit(static_argnums=[3])
def _generate_random_states(key, mean_activation, plaq_data, num_gen):
    shape = (plaq_data.shape[0], num_gen, plaq_data.shape[1])
    uniform = jax.random.uniform(key, shape=shape)
    flips = jnp.asarray(uniform < mean_activation, dtype=np.uint8)
    return (plaq_data[:, None, :] ^ flips).reshape((-1, plaq_data.shape[1]))


def _generate_random(
    parameters: Parameters,
    exp_data: list[list[RecoData]],
    mean_activation: list[np.ndarray],
    seed: int,
    logger: logging.Logger
) -> list[np.ndarray]:
    logger.info('Generating %d random samples for each of %d shots',
                parameters.skqd.num_gen, parameters.runtime.shots_exp)
    start = time.time()
    ndt = len(parameters.skqd.time_steps)
    nkr = parameters.skqd.num_krylov
    keys = jax.random.split(jax.random.key(seed), num=ndt * nkr).reshape((ndt, nkr))
    gen_states = [
        _generate_random_states(keys[idt][ikr], mean_activation[idt][ikr], exp_data[idt][ikr][1],
                                parameters.skqd.num_gen)
        for idt in range(ndt)
        for ikr in range(nkr)
    ]
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
    save_bits(out, 'sqd_states', sqd_states)
    out.create_dataset('energy', data=energy)
    out.create_dataset('eigvec', data=eigvec)
    if ham_proj is not None:
        group = out.create_group('ham_proj')
        group.create_dataset('data', data=ham_proj.data)
        group.create_dataset('indices', data=ham_proj.indices)
        group.create_dataset('indptr', data=ham_proj.indptr)


def diagonalize(
    parameters: Parameters,
    mode: str,  # 'cr' or 'rn'
    logger: Optional[logging.Logger] = None
) -> float:
    """Project and diagonalize the Hamiltonian with configuration recovery or random bitflips."""
    logger = logger or logging.getLogger(__name__)

    group_name = f'skqd_{mode}'
    saved_result = check_saved_result(parameters, group_name)
    if saved_result:
        logger.info('There is already an SKQD result saved in the file.')
        return saved_result[1]

    exp_data = load_reco(parameters, etype='exp')
    dual_lattice = make_dual_lattice(parameters)
    hamiltonian = dual_lattice.make_hamiltonian(parameters.lgt.plaquette_energy)
    num_plaquettes = dual_lattice.num_plaquettes

    if mode == 'cr':
        crbm_models = _prepare_data_and_models(parameters, exp_data, logger)
    else:
        mean_activation = _prepare_mean_activation(parameters, logger)

    energies = []
    subspace_dims = []
    relevant_states = None
    max_size = None
    is_last = False
    for it in range(parameters.skqd.max_iterations):
        logger.info('SKQD iteration %d', it)
        is_last = it == parameters.skqd.max_iterations - 1

        if mode == 'cr':
            states_list = _generate_cr(parameters, exp_data, crbm_models, logger)
        else:
            states_list = _generate_random(parameters, exp_data, mean_activation, 1234 + it, logger)

        if relevant_states is not None:
            states_list.append(relevant_states)
        states = np.concatenate(states_list, axis=0)
        states = np.unpackbits(uniquify_states(states), axis=1)[:, :num_plaquettes]
        for fname in parameters.skqd.extensions:
            states = extensions[fname](states, dual_lattice)

        # For small-scale problems, save compilation time by using padded fixed-size arrays
        if max_size is None and states.shape[0] < 1_000_000:
            max_size = int(states.shape[0] * 1.2)
            logger.info('Set maximum array size to %d', max_size)
        if max_size and (excess := states.shape[0] - max_size) > 0:
            max_size += excess * (parameters.skqd.max_iterations - it)
            logger.info('Updated maximum array size to %d', max_size)

        logger.info('Diagonalizing the Hamiltonian projected onto %d states..', states.shape[0])
        start = time.time()
        sqd_result = sqd(hamiltonian, states, states_size=max_size, return_hproj=is_last)
        energy, eigvec, sqd_states = sqd_result[:3]
        logger.info('Projection and diagonalization took %.2f seconds. Current energy %.4f',
                    time.time() - start, energy)

        energies.append(energy)
        subspace_dims.append(sqd_states.shape[0])

        if it > 0 and not is_last:
            if (energies[-2] - energies[-1] < parameters.skqd.delta_e
                    or subspace_dims[-1] >= parameters.skqd.max_subspace_dim):
                states_p = np.packbits(states, axis=1)
                sqd_result += (make_hproj(hamiltonian, states_p),)
                is_last = True
        if is_last:
            break

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
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--gpus', help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode')
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

    gen_modes = options.mode.split(',')

    if options.mpi:
        from mpi4py import MPI  # pylint: disable=no-name-in-module
        comm = MPI.COMM_WORLD
        gen_modes = [gen_modes[comm.Get_rank()]]

    for gen_mode in gen_modes:
        diagonalize(params, gen_mode)
