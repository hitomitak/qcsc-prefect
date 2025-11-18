"""Train the CRBM for configuration recovery."""
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


def check_model(
    parameters: Parameters,
    istep: int
) -> bool:
    path = Path(parameters.pkgpath) / 'crbm' / f'step{istep}.h5'
    return os.path.exists(path)


def load_model(
    parameters: Parameters,
    istep: int,
    jax_device_id: Optional[int] = None
) -> tuple[ConditionalRBM, dict[str, np.ndarray]]:
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
    ref_data: list[tuple[np.ndarray, np.ndarray]],
    train_fn: Callable,
    return_models: bool = True,
    logger: Optional[logging.Logger] = None
) -> list[ConditionalRBM]:
    logger = logger or logging.getLogger(__name__)

    steps_to_train = []

    if return_models:
        models = []
        for istep in range(parameters.skqd.n_trotter_steps):
            try:
                device_id = istep % jax.device_count()
                model = load_model(parameters, istep, device_id)[0]
            except FileNotFoundError:
                models.append(None)
                steps_to_train.append(istep)
            else:
                models.append(model)
    else:
        models = None
        for istep in range(parameters.skqd.n_trotter_steps):
            if not check_model(parameters, istep):
                steps_to_train.append(istep)

    if not steps_to_train:
        logger.info('All models already trained')
        return models

    logger.info('Training CRBMs for Trotter steps %s', steps_to_train)

    trained_models = train_fn(steps_to_train, ref_data)
    if return_models:
        for istep, model in trained_models.items():
            models[istep] = model

    return models


def train_generator(
    parameters: Parameters,
    ref_data: list[tuple[np.ndarray, np.ndarray]],
    logger: Optional[logging.Logger] = None
) -> list[ConditionalRBM]:
    # def train_fn_threading(steps_to_train, _):
    #     def train_on_device(istep, step_data, device_id):
    #         with jax.default_device(jax.devices()[device_id]):
    #             return train_step_model(istep, step_data[0], step_data[1], parameters.crbm)

    #     models = {}
    #     with ThreadPoolExecutor(jax.device_count()) as executor:
    #         futures = []
    #         for istep in steps_to_train:
    #             device_id = istep % jax.device_count()
    #             futures.append(
    #                 executor.submit(train_on_device, istep, ref_data[istep], device_id)
    #             )

    #     for istep, future in zip(steps_to_train, futures):
    #         model, records = future.result()
    #         save_model(parameters, istep, model, records)
    #         models[istep] = model

    #     return models

    def train_fn_subprocess(steps_to_train, _):
        def run_script(istep, idev):
            cmd = [
                sys.executable,
                pathlib.Path(__file__),
                parameters.pkgpath,
                f'{istep}',
                '--gpu', f'{idev}',  # train on idev
            ]
            proc = subprocess.run(cmd, capture_output=True, check=True, text=True)
            for txt, stream in zip([proc.stdout, proc.stderr], [sys.stdout, sys.stderr]):
                stream.write(txt)
                stream.flush()

            # Load the model onto istep % ndev (instead of idev)
            model, _ = load_model(parameters, istep, istep % jax.device_count())
            return model

        models = {}
        with ThreadPoolExecutor(jax.device_count()) as executor:
            futures = []
            for iproc, istep in enumerate(steps_to_train):
                idev = iproc % jax.device_count()
                futures.append(executor.submit(run_script, istep, idev))

        for istep, future in zip(steps_to_train, futures):
            model = future.result()
            models[istep] = model

        return models

    return train_generator_flow(parameters, ref_data, train_fn_subprocess, True, logger=logger)


def train_step_model(
    parameters: Parameters,
    istep: int,
    vtx_data: np.ndarray,
    plaq_data: np.ndarray,
    logger: Optional[logging.Logger] = None
):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('pkgpath')
    parser.add_argument('istep', type=int)
    parser.add_argument('--gpu')
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    with open(Path(options.pkgpath) / 'parameters.json', 'r', encoding='utf-8') as src:
        params = Parameters.model_validate_json(src.read())

    vdata, pdata = load_reco(params, etype='ref', istep=options.istep)
    train_step_model(params, options.istep, vdata, pdata)
