"""Train the CRBM for configuration recovery."""
import os
import argparse
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx
from skqd_z2lgt.crbm import ConditionalRBM
from skqd_z2lgt.train_crbm import DefaultCallback, make_l2_loss_fn, cd_meanloss, train_crbm

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('train_crbm')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('istep', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--out')
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

    with h5py.File(options.filename, 'r', swmr=True) as source:
        configuration = {}
        for key in source['configuration'].keys():
            record = source[f'configuration/{key}'][()]
            if isinstance(record, bytes):
                record = record.decode()
            configuration[key] = record

        num_plaq = source['data/num_plaq'][()]
        num_vtx = source['data/num_vtx'][()]
        ref_vtx_data = np.unpackbits(source['data/ref_vtx_data'], axis=-1)[..., :num_vtx]
        ref_plaq_data = np.unpackbits(source['data/ref_plaq_data'], axis=-1)[..., :num_plaq]
        train_u = ref_vtx_data[options.istep, :80_000]
        train_v = ref_plaq_data[options.istep, :80_000]
        test_u = ref_vtx_data[options.istep, 80_000:]
        test_v = ref_plaq_data[options.istep, 80_000:]

    mean_activation = np.mean(train_v, axis=0)
    mean_activation = np.where(np.isclose(mean_activation, 0.), 1.e-6, mean_activation)
    bias_init = np.log(mean_activation / (1. - mean_activation))

    rngs = nnx.Rngs(params=0, sample=1)
    model = ConditionalRBM(num_vtx, num_plaq, options.num_h, rngs=rngs)
    model.bias_v.value = bias_init
    model.bias_h.value = jnp.full(options.num_h,
                                  np.log(options.init_h_sparsity / (1. - options.init_h_sparsity)))
    model.pregenerate = True

    LOG.info('Start training model for step %d', options.istep)

    loss_fn = make_l2_loss_fn(cd_meanloss, options.l2w_weights, options.l2w_biases)
    best_model, records = train_crbm(model, train_u, train_v, test_u, test_v,
                                     options.batch_size, options.num_epochs, loss_fn,
                                     lr=options.learning_rate, rtol=options.rtol,
                                     callback=DefaultCallback())

    out_filename = options.out or options.filename
    groupname = f'crbm_step{options.istep}'  # pylint: disable=invalid-name
    with h5py.File(out_filename, 'w-' if options.out else 'r+', libver='latest') as out:
        try:
            del out[groupname]
        except KeyError:
            pass
        group = out.create_group(f'crbm_step{options.istep}')
        best_model.save(group)
        group = group.create_group('records')
        for key, array in records.items():
            group.create_dataset(key, data=array)
