"""Train the CRBM for configuration recovery."""
from collections import namedtuple
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.train_crbm import DefaultCallback, make_l2_loss_fn, cd_meanloss, train_crbm
from skqd_z2lgt.parameters import Parameters
from skqd_z2lgt.tasks.preprocess import load_reco


def load_model(
    istep: int,
    source_filename: str,
    jax_device_id: Optional[int] = None
) -> tuple[ConditionalRBM, dict[str, np.ndarray]]:
    if jax_device_id is None:
        device = None
    else:
        device = jax.devices()[jax_device_id]

    with h5py.File(source_filename, 'r', libver='latest') as out:
        group = out[f'crbm/step{istep}']
        with jax.default_device(device):
            model = ConditionalRBM.load(group)
        records = {key: dataset[()] for key, dataset in group['records'].items()}
    return model, records


def save_model(
    istep: int,
    model: ConditionalRBM,
    records: dict[str, np.ndarray],
    output_filename: str
):
    with h5py.File(output_filename, 'a', libver='latest') as out:
        groupname = f'crbm/step{istep}'
        group = out.create_group(groupname)
        model.save(group)
        group = group.create_group('records')
        for key, array in records.items():
            group.create_dataset(key, data=array)


def train_generator_flow(
    parameters: Parameters,
    ref_data: list[tuple[np.ndarray, np.ndarray]],
    train_fn: Callable,
    logger: Optional[logging.Logger] = None
) -> list[ConditionalRBM]:
    logger = logger or logging.getLogger(__name__)

    models = []
    steps_to_train = []
    for istep in range(parameters.skqd.n_trotter_steps):
        try:
            model = load_model(istep, parameters.output_filename)[0]
        except KeyError:
            models.append(None)
            steps_to_train.append(istep)
        else:
            models.append(model)

    if not steps_to_train:
        logger.info('All models already trained')
        return models

    logger.info('Training CRBMs for Trotter steps %s', steps_to_train)

    trained_models = train_fn(steps_to_train, ref_data)
    for istep, model in trained_models.items():
        models[istep] = model

    return models


def train_generator(
    parameters: Parameters,
    ref_data: list[tuple[np.ndarray, np.ndarray]],
    logger: Optional[logging.Logger] = None
) -> list[ConditionalRBM]:
    conf = parameters.crbm
    model_params = {'num_h': conf.num_h, 'init_h_sparsity': conf.init_h_sparsity}
    train_params = {'l2w_weights': conf.l2w_weights, 'l2w_biases': conf.l2w_biases,
                    'batch_size': conf.train_batch_size, 'learning_rate': conf.learning_rate,
                    'num_epochs': conf.num_epochs, 'rtol': conf.rtol}

    def train_fn(steps_to_train, ref_data):
        def train_on_device(istep, step_data, device_id):
            with jax.default_device(jax.devices()[device_id]):
                return train_step_model(istep, step_data[0], step_data[1], model_params,
                                        train_params)

        models = {}
        with ThreadPoolExecutor(jax.device_count()) as executor:
            futures = []
            for istep in steps_to_train:
                device_id = istep % jax.device_count()
                futures.append(executor.submit(train_on_device, istep, ref_data[istep], device_id))

        for istep, future in zip(steps_to_train, futures):
            model, records = future.result()
            save_model(istep, model, records, parameters.output_filename)
            models[istep] = model

        return models

    train_generator_flow(parameters, ref_data, train_fn, logger)


def train_step_model(
    istep: int,
    vtx_data: np.ndarray,
    plaq_data: np.ndarray,
    model_params: dict[str, Any],
    train_params: dict[str, Any],
    out_filename: Optional[str] = None,
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

    rngs = nnx.Rngs(params=0, sample=1)
    model = ConditionalRBM(num_vtx, num_plaq, model_params['num_h'], rngs=rngs)
    model.bias_v.value = bias_init
    bias_h_val = np.log(model_params['init_h_sparsity'] / (1. - model_params['init_h_sparsity']))
    model.bias_h.value = jnp.full(model_params['num_h'], bias_h_val)

    model.pregenerate = True

    logger.info('Start training model for step %d', istep)

    loss_fn = make_l2_loss_fn(cd_meanloss, train_params['l2w_weights'], train_params['l2w_biases'])
    best_model, records = train_crbm(model, train_u, train_v, test_u, test_v,
                                     train_params['batch_size'], train_params['num_epochs'],
                                     loss_fn, lr=train_params['learning_rate'],
                                     rtol=train_params['rtol'], callback=DefaultCallback())
    if out_filename:
        save_model(istep, best_model, records, out_filename)
    return best_model, records


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('istep', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--out-filename')
    parser.add_argument('--num-h', type=int, default=256)
    parser.add_argument('--l2w-weights', type=float, default=1.)
    parser.add_argument('--l2w-biases', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--init-h-sparsity', type=float, default=0.01)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--rtol', type=float)
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    jax.config.update('jax_enable_x64', True)

    mparams = {'num_h': options.num_h, 'init_h_sparsity': options.init_h_sparsity}
    tparams = {'l2w_weights': options.l2w_weights, 'l2w_biases': options.l2w_biases,
               'batch_size': options.batch_size, 'learning_rate': options.learning_rate,
               'num_epochs': options.num_epochs, 'rtol': options.rtol}
    params = namedtuple('OutputFileName', ['output_filename'])(options.filename)
    vdata, pdata = load_reco(params, etype='ref', istep=options.itep)
    train_step_model(options.istep, vdata, pdata, mparams, tparams,
                     out_filename=options.out_filename)
