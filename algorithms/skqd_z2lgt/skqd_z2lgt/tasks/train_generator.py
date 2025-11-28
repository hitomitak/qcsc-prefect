"""Train the CRBM for configuration recovery."""
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
import logging
import pathlib
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.train_crbm import DefaultCallback, make_l2_loss_fn, cd_meanloss, train_crbm
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.preprocess import load_reco


def check_models(
    parameters: Parameters
) -> list[int]:
    """Check existence of saved models without loading."""
    missing_isteps = []
    dirpath = Path(parameters.pkgpath) / 'crbm'
    for istep in range(parameters.skqd.n_trotter_steps):
        if not os.path.exists(dirpath / f'step{istep}.h5'):
            missing_isteps.append(istep)
    return missing_isteps


def load_model(
    parameters: Parameters,
    istep: int,
    jax_device_id: Optional[int] = None
) -> tuple[ConditionalRBM, dict[str, np.ndarray]]:
    """Construct CRBM models from saved weights."""
    if jax_device_id is None:
        device = None
    else:
        device = jax.devices()[jax_device_id]

    path = Path(parameters.pkgpath) / 'crbm' / f'step{istep}.h5'
    with h5py.File(path, 'r', libver='latest') as source:
        with jax.default_device(device):
            model = ConditionalRBM.load(source)
        records = {key: dataset[()] for key, dataset in source['records'].items()}
    return model, records


def save_model(
    parameters: Parameters,
    istep: int,
    model: ConditionalRBM,
    records: dict[str, np.ndarray]
):
    """Save the weight parameters of the CRBM model into file."""
    path = Path(parameters.pkgpath) / 'crbm' / f'step{istep}.h5'
    try:
        os.makedirs(path.parent)
    except FileExistsError:
        pass

    with h5py.File(path, 'w', libver='latest') as source:
        model.save(source)
        group = source.create_group('records')
        for key, array in records.items():
            group.create_dataset(key, data=array)


def train_generator_flow(
    parameters: Parameters,
    train_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """General flow for training CRBM models."""
    logger = logger or logging.getLogger(__name__)

    steps_to_train = check_models(parameters)
    if not steps_to_train:
        logger.info('All models already trained')
        return

    logger.info('Training CRBMs for Trotter steps %s', steps_to_train)
    train_fn(steps_to_train)


def train_generator(
    parameters: Parameters,
    logger: Optional[logging.Logger] = None
):
    """Standalone training function."""
    # def train_fn(steps_to_train):
    #     def train_on_device(istep, device_id):
    #         vdata, pdata = load_reco(parameters, etype='ref', istep=istep)
    #         with jax.default_device(jax.devices()[device_id]):
    #             return train_step_model(istep, vdata, pdata, parameters.crbm)

    #     with ThreadPoolExecutor(jax.device_count()) as executor:
    #         futures = []
    #         for istep in steps_to_train:
    #             device_id = istep % jax.device_count()
    #             futures.append(
    #                 executor.submit(train_on_device, istep, device_id)
    #             )

    #     for istep, future in zip(steps_to_train, futures):
    #         model, records = future.result()
    #         save_model(parameters, istep, model, records)

    def train_fn(steps_to_train):
        def run_script(istep, igpu):
            cmd = [
                sys.executable,
                str(pathlib.Path(__file__)),
                parameters.pkgpath,
                '--istep', f'{istep}',
                '--gpu', f'{igpu}',  # train on igpu
            ]
            proc = subprocess.run(cmd, capture_output=True, check=True, text=True)
            for txt, stream in zip([proc.stdout, proc.stderr], [sys.stdout, sys.stderr]):
                stream.write(txt)
                stream.flush()

        with ThreadPoolExecutor(jax.device_count()) as executor:
            if (cvd := os.getenv('CUDA_VISIBLE_DEVICES')):
                gpus = list(map(int, cvd.split(',')))
            else:
                gpus = list(range(jax.device_count()))
            futures = []
            for iproc, istep in enumerate(steps_to_train):
                igpu = gpus[iproc % len(gpus)]
                futures.append(executor.submit(run_script, istep, igpu))

        wait(futures)

    train_generator_flow(parameters, train_fn, logger=logger)


def train_step_model(
    parameters: Parameters,
    istep: int,
    vtx_data: np.ndarray,
    plaq_data: np.ndarray,
    logger: Optional[logging.Logger] = None
):
    """Train a single CRBM model."""
    logger = logger or logging.getLogger(__name__)

    train_u = vtx_data[:80_000]
    train_v = plaq_data[:80_000]
    test_u = vtx_data[80_000:]
    test_v = plaq_data[80_000:]
    mean_activation = np.mean(train_v, axis=0)
    mean_activation = np.where(np.isclose(mean_activation, 0.), 1.e-6, mean_activation)
    bias_init = np.log(mean_activation / (1. - mean_activation))

    num_vtx = vtx_data.shape[-1]
    num_plaq = plaq_data.shape[-1]

    crbm_params = parameters.crbm

    rngs = nnx.Rngs(params=0, sample=1)
    model = ConditionalRBM(num_vtx, num_plaq, crbm_params.num_h, rngs=rngs)
    model.bias_v.value = bias_init
    bias_h_val = np.log(crbm_params.init_h_sparsity / (1. - crbm_params.init_h_sparsity))
    model.bias_h.value = jnp.full(crbm_params.num_h, bias_h_val)

    model.pregenerate = True

    logger.info('Start training model for step %d', istep)

    loss_fn = make_l2_loss_fn(cd_meanloss, crbm_params.l2w_weights, crbm_params.l2w_biases)
    best_model, records = train_crbm(model, train_u, train_v, test_u, test_v,
                                     crbm_params.train_batch_size, crbm_params.num_epochs,
                                     loss_fn, lr=crbm_params.learning_rate,
                                     rtol=crbm_params.rtol, callback=DefaultCallback())
    save_model(parameters, istep, best_model, records)
    return best_model, records


if __name__ == '__main__':
    import argparse
    from mpi4py import MPI  # pylint: disable=no-name-in-module

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--istep', type=int, nargs='+')
    parser.add_argument('--gpu')
    parser.add_argument('--log-level', default='INFO')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()),
                        stream=sys.stdout)
    logging.getLogger('jax').setLevel(logging.WARNING)

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    with open(Path(options.pkgpath) / 'parameters.json', 'r', encoding='utf-8') as src:
        params = Parameters.model_validate_json(src.read())

    if options.mpi:
        if options.istep is None:
            isteps = list(range(params.skqd.n_trotter_steps))
        else:
            isteps = options.istep

        comm = MPI.COMM_WORLD
        ist = isteps[comm.Get_rank()]
    else:
        ist = options.istep[0]

    vdata, pdata = load_reco(params, etype='ref', istep=ist)
    train_step_model(params, ist, vdata, pdata)
