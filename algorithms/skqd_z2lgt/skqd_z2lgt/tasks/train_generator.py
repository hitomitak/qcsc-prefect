"""Train the CRBM for configuration recovery."""
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
import logging
import subprocess
from pathlib import Path
from itertools import product
from typing import Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.train_crbm import SuccessRateCallback, make_l2_loss_fn, cd_meanloss, train_crbm
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.preprocess import load_reco


def save_model(
    parameters: Parameters,
    model: ConditionalRBM,
    records: dict[str, np.ndarray],
    idt: int,
    ikrylov: int,
    logger: Optional[logging.Logger] = None
):
    """Save the weight parameters of the CRBM model into file."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Saving CRBM model for idt=%d ikrylov=%d', idt, ikrylov)

    path = Path(parameters.pkgpath) / 'crbm' / f'dt{idt}_k{ikrylov}.h5'
    os.makedirs(path.parent, exist_ok=True)
    with h5py.File(path, 'w', libver='latest') as out:
        model.save(out)
        group = out.create_group('records')
        for key, array in records.items():
            group.create_dataset(key, data=array)


def check_model(
    parameters: Parameters,
    idt: int,
    ikrylov: int,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Check existence of a saved model without loading."""
    logger = logger or logging.getLogger(__name__)
    path = Path(parameters.pkgpath) / 'crbm' / f'dt{idt}_k{ikrylov}.h5'
    if os.path.exists(path):
        logger.info('CRBM model for idt=%d ikrylov=%d already exists.', idt, ikrylov)
        return True
    return False


def load_model(
    parameters: Parameters,
    idt: int,
    ikrylov: int,
    compile_for: Optional[tuple[int, int]] = None,
    jax_device_id: Optional[int] = None
) -> ConditionalRBM:
    """Construct CRBM models from saved weights."""
    if jax_device_id is None:
        device = None
    else:
        device = jax.devices()[jax_device_id]

    path = Path(parameters.pkgpath) / 'crbm' / f'dt{idt}_k{ikrylov}.h5'
    with h5py.File(path, 'r', libver='latest') as source:
        with jax.default_device(device):
            model = ConditionalRBM.load(source)
            if compile_for:
                model.sample(
                    jnp.zeros((compile_for[0], model.weights_vu.shape[1]), dtype=np.uint8),
                    size=compile_for[1]
                )
    return model


def train_generator_flow(
    parameters: Parameters,
    train_fn: Callable,
    logger: Optional[logging.Logger] = None
):
    """General flow for training CRBM models."""
    logger = logger or logging.getLogger(__name__)

    tasks = []
    for idt in range(len(parameters.skqd.time_steps)):
        for ikrylov in range(1, parameters.skqd.num_krylov + 1):
            if not check_model(parameters, idt, ikrylov, logger=logger):
                tasks.append((idt, ikrylov))

    if not tasks:
        logger.info('All models already trained')
        return

    logger.info('Training %d CRBMs', len(tasks))
    train_fn(tasks)


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

    def train_fn(tasks):
        def run_script(idt, ikrylov, igpu):
            cmd = [
                sys.executable, str(Path(__file__)),
                parameters.pkgpath,
                '--idt', f'{idt}',
                '--ikrylov', f'{ikrylov}',
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
            for iproc, (idt, ikrylov) in enumerate(tasks):
                igpu = gpus[iproc % len(gpus)]
                futures.append(executor.submit(run_script, idt, ikrylov, igpu))

        wait(futures)

    train_generator_flow(parameters, train_fn, logger=logger)


def train_single_model(
    parameters: Parameters,
    idt: int,
    ikrylov: int,
    logger: Optional[logging.Logger] = None
):
    """Train a single CRBM model."""
    logger = logger or logging.getLogger(__name__)
    logger.info('Training CRBM model for time interval %f, Krylov vector %d',
                parameters.skqd.time_steps[idt], ikrylov)

    vtx_data, plaq_data = load_reco(parameters, etype='ref', idt=idt, ikrylov=ikrylov)

    num_train = vtx_data.shape[0] // 10 * 8
    train_u = vtx_data[:num_train]
    train_v = plaq_data[:num_train]
    test_u = vtx_data[num_train:]
    test_v = plaq_data[num_train:]
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

    loss_fn = make_l2_loss_fn(cd_meanloss, crbm_params.l2w_weights, crbm_params.l2w_biases)
    best_model, records = train_crbm(model, train_u, train_v, test_u, test_v,
                                     crbm_params.train_batch_size, crbm_params.num_epochs,
                                     loss_fn, lr=crbm_params.learning_rate,
                                     callback=SuccessRateCallback())
    save_model(parameters, best_model, records, idt, ikrylov, logger=logger)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--idt')
    parser.add_argument('--ikrylov')
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

    if options.idt:
        idts = list(map(int, options.idt.split(',')))
        ikrylovs = list(map(int, options.ikrylov.split(',')))
        if len(idts) != len(ikrylovs):
            raise ValueError('Lengths of idt and ikrylov lists do not match')
        task_specs = list(zip(idts, ikrylovs))
    else:
        idts = list(range(len(params.skqd.time_steps)))
        ikrylovs = list(range(1, params.skqd.num_krylov + 1))
        task_specs = list(product(idts, ikrylovs))

    if options.mpi:
        from mpi4py import MPI  # pylint: disable=no-name-in-module
        comm = MPI.COMM_WORLD
        task_specs = [task_specs[comm.Get_rank()]]

    for task_spec in task_specs:
        train_single_model(params, *task_spec)
