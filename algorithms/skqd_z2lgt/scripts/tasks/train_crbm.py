"""Train the CRBM for configuration recovery."""
import os
import argparse
import logging
from typing import Any, Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.train_crbm import DefaultCallback, make_l2_loss_fn, cd_meanloss, train_crbm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('train_crbm')


def main(
    filename: str,
    istep: int,
    model_params: dict[str, Any],
    train_params: dict[str, Any],
    out_filename: Optional[str] = None
):
    with h5py.File(filename, 'r', swmr=True) as source:
        configuration = {}
        for key in source['configuration'].keys():
            record = source[f'configuration/{key}'][()]
            if isinstance(record, bytes):
                record = record.decode()
            configuration[key] = record

        group = source[f'ref_step{istep}']
        num_plaq = group['num_plaq'][()]
        num_vtx = group['num_vtx'][()]
        vtx_data = np.unpackbits(group['vtx_data'], axis=-1)[..., :num_vtx]
        plaq_data = np.unpackbits(group['plaq_data'], axis=-1)[..., :num_plaq]
        train_u = vtx_data[:80_000]
        train_v = plaq_data[:80_000]
        test_u = vtx_data[80_000:]
        test_v = plaq_data[80_000:]

    mean_activation = np.mean(train_v, axis=0)
    mean_activation = np.where(np.isclose(mean_activation, 0.), 1.e-6, mean_activation)
    bias_init = np.log(mean_activation / (1. - mean_activation))

    rngs = nnx.Rngs(params=0, sample=1)
    model = ConditionalRBM(num_vtx, num_plaq, model_params['num_h'], rngs=rngs)
    model.bias_v.value = bias_init
    bias_h_val = np.log(model_params['init_h_sparsity'] / (1. - model_params['init_h_sparsity']))
    model.bias_h.value = jnp.full(model_params['num_h'], bias_h_val)

    model.pregenerate = True

    LOG.info('Start training model for step %d', istep)

    loss_fn = make_l2_loss_fn(cd_meanloss, train_params['l2w_weights'], train_params['l2w_biases'])
    best_model, records = train_crbm(model, train_u, train_v, test_u, test_v,
                                     train_params['batch_size'], train_params['num_epochs'],
                                     loss_fn, lr=train_params['learning_rate'],
                                     rtol=train_params['rol'], callback=DefaultCallback())

    file_mode = 'w' if out_filename else 'r+'
    out_filename = out_filename or filename
    groupname = f'crbm_step{istep}'  # pylint: disable=invalid-name
    with h5py.File(out_filename, file_mode, libver='latest') as out:
        try:
            del out[groupname]
        except KeyError:
            pass
        group = out.create_group(f'crbm_step{istep}')
        best_model.save(group)
        group = group.create_group('records')
        for key, array in records.items():
            group.create_dataset(key, data=array)


if __name__ == '__main__':
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
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_enable_x64', True)

    mparams = {'num_h': options.num_h, 'init_h_sparsity': options.init_h_sparsity}
    tparams = {'l2w_weights': options.l2w_weights, 'l2w_biases': options.l2w_biases,
               'batch_size': options.batch_size, 'learning_rate': options.learning_rate,
               'num_epochs': options.num_epochs, 'rtol': options.rtol}
    main(options.filename, options.istep, mparams, tparams, out_filename=options.out_filename)
