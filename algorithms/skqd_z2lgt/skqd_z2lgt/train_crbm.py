"""Classes and routines for training the CRBM."""
from collections.abc import Callable
from itertools import product
import logging
from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from .crbm import ConditionalRBM

LOG = logging.getLogger(__name__)


class BaseCallback(nnx.Module):
    """Base class for CRBM training callback."""
    def init_records(self) -> dict[str, Any]:
        """Create a container to export the records to."""
        return {}

    def train_step(
        self,
        model: ConditionalRBM,
        u_batch: jax.Array,
        v_batch: jax.Array,
        loss: jax.Array,
        grads: jax.Array
    ):
        """Callback within train_step."""

    def train_eval(
        self,
        model: ConditionalRBM,
        iepoch: int,
        ibatch: int,
        records: dict[str, Any]
    ):
        """Callback at evaluation during training."""

    def test(
        self,
        model: ConditionalRBM,
        test_u: jax.Array,
        test_v: jax.Array,
        iepoch: int,
        records: dict[str, Any]
    ):
        """Callback for per-epoch tests."""


class DefaultCallback(BaseCallback):
    """Default callback module for recording loss and free energy histories."""
    def __init__(self, eval_every: int = 10):
        self.metrics = nnx.metrics.MultiMetric(
            loss=nnx.metrics.Average('loss'),
            free_energy=nnx.metrics.Average('free_energy')
        )
        self.eval_every = eval_every

    def init_records(self) -> dict[str, Any]:
        return {'_'.join(comb): []
                for comb in product(['train', 'test'], self.metrics._metric_names)}

    def as_arrays(self, records: dict[str, list]):
        for key, value in records.items():
            records[key] = np.array(value)

    @nnx.jit
    def train_step(
        self,
        model: ConditionalRBM,
        u_batch: jax.Array,
        v_batch: jax.Array,
        loss: jax.Array,
        grads: jax.Array
    ):
        """Callback within train_step."""
        free_energy = jnp.mean(model.free_energy(u_batch, v_batch))
        self.metrics.update(loss=loss, free_energy=free_energy)

    def train_eval(
        self,
        model: ConditionalRBM,
        iepoch: int,
        ibatch: int,
        records: dict[str, Any]
    ):
        """Callback at evaluation during training."""
        if (ibatch + 1) % self.eval_every != 0:
            return
        for metric, value in self.metrics.compute().items():
            records[f'train_{metric}'].append(float(value))
        self.metrics.reset()

    @nnx.jit
    def _test_update(
        self,
        model: ConditionalRBM,
        test_u: jax.Array,
        test_v: jax.Array
    ):
        vhat = model.percloss_states(test_u)
        loss = loss_fn(model, test_u, test_v, vhat)
        free_energy = jnp.mean(model.free_energy(test_u, test_v))
        self.metrics.update(loss=loss, free_energy=free_energy)

    def test(
        self,
        model: ConditionalRBM,
        test_u: jax.Array,
        test_v: jax.Array,
        iepoch: int,
        records: dict[str, Any]
    ):
        self._test_update(model, test_u, test_v)
        for metric, value in self.metrics.compute().items():
            records[f'test_{metric}'].append(float(value))
        self.metrics.reset()


@nnx.jit
def loss_fn(
    model: ConditionalRBM,
    u_state: jax.Array,
    v_state: jax.Array,
    vhat_state: jax.Array
):
    return jnp.mean(model.percloss(u_state, v_state, vhat_state))


grad_fn = nnx.jit(nnx.value_and_grad(loss_fn))


@nnx.jit
def train_step(
    model: ConditionalRBM,
    u_batch: jax.Array,
    v_batch: jax.Array,
    optimizer: nnx.optimizer.Optimizer,
    callback: BaseCallback
):
    vhat_batch = model.percloss_states(u_batch)
    loss, grads = grad_fn(model, u_batch, v_batch, vhat_batch)
    callback.train_step(model, u_batch, v_batch, loss, grads)
    optimizer.update(model, grads)


def train_crbm(
    model: ConditionalRBM,
    train_dataset: np.ndarray,
    test_dataset: np.ndarray,
    batch_size: int,
    num_epochs: int,
    optax_fn: Optional[Callable] = None,
    seed: int = 0,
    callback: Optional[BaseCallback] = None
):
    optax_fn = optax_fn or optax.adamw(learning_rate=0.005)
    optimizer = nnx.Optimizer(model, optax_fn, wrt=nnx.Param)
    callback = callback or BaseCallback()
    records = callback.init_records()

    rng = np.random.default_rng(seed)
    num_batches = train_dataset.shape[0] // batch_size
    num_u = model.weights_hu.shape[1]

    test_u = jax.device_put(test_dataset[:, :num_u])
    test_v = jax.device_put(test_dataset[:, num_u:])

    for iepoch in range(num_epochs):
        LOG.info('Starting epoch %d/%d', iepoch, num_epochs)
        sample_indices = np.arange(train_dataset.shape[0])
        rng.shuffle(sample_indices)
        samples_u = jax.device_put(train_dataset[sample_indices][:, :num_u])
        samples_v = jax.device_put(train_dataset[sample_indices][:, num_u:])

        start = 0
        for ibatch in range(num_batches):
            LOG.debug('Batch %d/%d', ibatch, num_batches)
            end = start + batch_size
            u_batch, v_batch = samples_u[start:end], samples_v[start:end]
            train_step(model, u_batch, v_batch, optimizer, callback)
            start = end
            callback.train_eval(model, iepoch, ibatch, records)

        callback.test(model, test_u, test_v, iepoch, records)

    return records
