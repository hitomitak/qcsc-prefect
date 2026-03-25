# pylint: disable=unused-argument
"""Classes and routines for training the CRBM."""
from collections.abc import Callable
import logging
from typing import Any, Optional
import numpy as np
import h5py
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
        loss: jax.Array
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
        loss: jax.Array,
        iepoch: int,
        records: dict[str, Any]
    ) -> bool:
        """Callback for per-epoch tests. Returns True when stop condition is met."""
        return False


class DefaultCallback(BaseCallback):
    """Default callback module for recording loss and free energy histories."""
    def __init__(
        self,
        eval_every: int = 100,
        train_metrics: Optional[dict] = None,
        test_metrics: Optional[dict] = None
    ):
        train_metrics = train_metrics or {}
        self.train_metrics = nnx.metrics.MultiMetric(
            loss=nnx.metrics.Average('loss'),
            free_energy=nnx.metrics.Average('free_energy'),
            **train_metrics
        )
        test_metrics = test_metrics or {}
        self.test_metrics = nnx.metrics.MultiMetric(
            loss=nnx.metrics.Average('loss'),
            free_energy=nnx.metrics.Average('free_energy'),
            **test_metrics
        )
        self.eval_every = eval_every

    def init_records(self) -> dict[str, Any]:
        records = {f'train_{name}': [] for name in self.train_metrics._metric_names}
        records |= {f'test_{name}': [] for name in self.test_metrics._metric_names}
        return records

    def as_arrays(self, records: dict[str, list]):
        for key, value in records.items():
            records[key] = np.array(value)

    @nnx.jit
    def train_step(
        self,
        model: ConditionalRBM,
        u_batch: jax.Array,
        v_batch: jax.Array,
        loss: jax.Array
    ):
        """Callback within train_step."""
        free_energy = jnp.mean(model.free_energy(u_batch, v_batch))
        updates = {'loss': loss, 'free_energy': free_energy}
        updates |= self._train_step_ext(model, u_batch, v_batch, updates)
        self.train_metrics.update(**updates)

    def _train_step_ext(self, model, u_batch, v_batch, updates):
        return {}

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
        for metric, value in self.train_metrics.compute().items():
            records[f'train_{metric}'].append(float(value))
        self.train_metrics.reset()

    @nnx.jit
    def _test_update(
        self,
        model: ConditionalRBM,
        test_u: jax.Array,
        test_v: jax.Array,
        loss: jax.Array
    ):
        free_energy = jnp.mean(model.free_energy(test_u, test_v))
        updates = {'loss': loss, 'free_energy': free_energy}
        updates |= self._test_update_ext(model, test_u, test_v, updates)
        self.test_metrics.update(**updates)

    def _test_update_ext(self, model, test_u, test_v, updates):
        return {}

    def test(
        self,
        model: ConditionalRBM,
        test_u: jax.Array,
        test_v: jax.Array,
        loss: jax.Array,
        iepoch: int,
        records: dict[str, Any]
    ) -> bool:
        self._test_ext(model, test_u, test_v, loss, iepoch)
        self._test_update(model, test_u, test_v, loss)
        for metric, value in self.test_metrics.compute().items():
            records[f'test_{metric}'].append(float(value))
        self.test_metrics.reset()
        return self._stop_training(iepoch, records)

    def _test_ext(
        self,
        model: ConditionalRBM,
        test_u: jax.Array,
        test_v: jax.Array,
        loss: jax.Array,
        iepoch: int
    ):
        pass

    def _stop_training(self, iepoch: int, records: dict[str, Any]) -> bool:
        if iepoch < 5:
            return False
        recent = np.array(records['test_loss'][-5:])
        if np.nonzero(np.diff(recent) > 0.)[0].shape[0] < 2:
            return False
        mean = np.mean(recent)
        stddev = np.std(recent)
        return np.all(np.abs(recent - mean) < stddev * 2.)


class NLLCallback(DefaultCallback):
    """Callback with NLL calculation."""
    def __init__(self, eval_every=100):
        train_metrics = {'nll': nnx.metrics.Average('nll')}
        test_metrics = {'nll': nnx.metrics.Average('nll')}
        super().__init__(eval_every=eval_every, train_metrics=train_metrics,
                         test_metrics=test_metrics)

    @nnx.jit
    def _train_step_ext(self, model, u_batch, v_batch, updates):
        """Callback within train_step."""
        logz, norm = model.conditional_logz(u_batch)
        nll = updates['free_energy'] + jnp.mean(-norm + logz)
        return {'nll': nll}

    @nnx.jit
    def _test_update_ext(self, model, test_u, test_v, updates):
        return self._train_step_ext(model, test_u, test_v, updates)


class SuccessRateCallback(DefaultCallback):
    """Callback with recovery success rate calculation at test steps."""
    def __init__(self, eval_every=100):
        metrics = {f'success_{n}': nnx.metrics.Average(f'success_{n}')
                   for n in [5, 10, 20, 100]}
        super().__init__(eval_every=eval_every, test_metrics=metrics)

    @nnx.jit
    def _test_update_ext(self, model, test_u, test_v, updates):
        samples = model.sample(test_u, 100)
        test = jnp.all(jnp.equal(test_v[None, ...], samples), axis=-1)
        return {f'success_{n}': jnp.any(test[:n], axis=0).astype(int) for n in [5, 10, 20, 100]}

    def _stop_training(self, iepoch: int, records: dict[str, Any]) -> bool:
        """Stop the training when more than 4 out of past 10 epochs failed to improve the smoothed
        success rate in 100 samples."""
        if iepoch < 20:
            return False
        sr = records['test_success_100']
        nsr = len(sr)
        smoothed = np.mean([sr[i:nsr - 9 + i] for i in range(10)], axis=0)
        return np.count_nonzero(np.diff(smoothed[-10:]) < 0.) > 4


@nnx.jit
def cd_percloss(
    model: ConditionalRBM,
    u_state: jax.Array,
    v_state: jax.Array
):
    vhat_state = jax.lax.stop_gradient(model.percloss_states(u_state))
    return jnp.mean(model.percloss(u_state, v_state, vhat_state))


@nnx.jit
def cd_meanloss(
    model: ConditionalRBM,
    u_state: jax.Array,
    v_state: jax.Array
):
    vg_states = jax.lax.stop_gradient(model.sample(u_state, model.vhat_size))
    return jnp.mean(model.meanloss(u_state, v_state, vg_states))


@nnx.jit
def nll_loss(
    model: ConditionalRBM,
    u_state: jax.Array,
    v_state: jax.Array
):
    return jnp.mean(model.conditional_nll(u_state, v_state))


def l2_regularization(model, l2w_weights, l2w_biases):
    """L2 regularization term."""
    return (
        l2w_weights * (
            jnp.mean(jnp.square(model.weights_vu))
            + jnp.mean(jnp.square(model.weights_hu))
            + jnp.mean(jnp.square(model.weights_hv))
        )
        + l2w_biases * (
            jnp.mean(jnp.square(model.bias_v))
            + jnp.mean(jnp.square(model.bias_h))
        )
    )


def make_l2_loss_fn(loss_fn: Callable, l2w_weights: float, l2w_biases: float):
    """Construct a loss function with L2 regularization."""
    def fn(
        model: ConditionalRBM,
        u_state: jax.Array,
        v_state: jax.Array
    ):
        return loss_fn(model, u_state, v_state) + l2_regularization(model, l2w_weights, l2w_biases)

    return nnx.jit(fn)


def train_crbm(
    model: ConditionalRBM,
    train_u: np.ndarray,
    train_v: np.ndarray,
    test_u: np.ndarray,
    test_v: np.ndarray,
    batch_size: int,
    num_epochs: int,
    loss_fn: Callable[[ConditionalRBM, jax.Array, jax.Array], jax.Array],
    lr: float = 0.001,
    optax_fn: Optional[Callable] = None,
    seed: int = 0,
    callback: Optional[BaseCallback] = None,
    records: Optional[dict[str, Any]] = None
):
    if isinstance(loss_fn, tuple) and loss_fn[0] == 'cd':
        l2w_weights, l2w_biases = loss_fn[1:]

        loss_fn = make_l2_loss_fn(cd_percloss, l2w_weights, l2w_biases)

        @nnx.jit
        def _loss_fn(
            model: ConditionalRBM,
            u_state: jax.Array,
            v_state: jax.Array,
            vhat_state: jax.Array
        ):
            loss = jnp.mean(model.percloss(u_state, v_state, vhat_state))
            loss += l2_regularization(model, l2w_weights, l2w_biases)
            return loss

        @nnx.jit
        def train_step(
            model: ConditionalRBM,
            u_batch: jax.Array,
            v_batch: jax.Array,
            optimizer: nnx.optimizer.Optimizer,
            callback: BaseCallback
        ):
            grad_fn = nnx.value_and_grad(_loss_fn)
            vhat_batch = model.percloss_states(u_batch)
            loss, grads = grad_fn(model, u_batch, v_batch, vhat_batch)
            callback.train_step(model, u_batch, v_batch, loss)
            optimizer.update(model, grads)
            return loss

    else:
        @nnx.jit
        def train_step(
            model: ConditionalRBM,
            u_batch: jax.Array,
            v_batch: jax.Array,
            optimizer: nnx.optimizer.Optimizer,
            callback: BaseCallback
        ):
            grad_fn = nnx.value_and_grad(loss_fn)
            loss, grads = grad_fn(model, u_batch, v_batch)
            callback.train_step(model, u_batch, v_batch, loss)
            optimizer.update(model, grads)
            return loss

    optax_fn = optax_fn or optax.adamw(learning_rate=lr)
    optimizer = nnx.Optimizer(model, optax_fn, wrt=nnx.Param)
    callback = callback or BaseCallback()
    records = records or callback.init_records()

    rng = np.random.default_rng(seed)
    num_batches = train_u.shape[0] // batch_size

    test_u = jax.device_put(test_u)
    test_v = jax.device_put(test_v)

    epoch_losses = []
    best_model_snapshot = None
    for iepoch in range(num_epochs):
        LOG.info('Starting epoch %d/%d', iepoch, num_epochs)
        sample_indices = np.arange(train_u.shape[0])
        rng.shuffle(sample_indices)
        samples_u = jax.device_put(train_u[sample_indices])
        samples_v = jax.device_put(train_v[sample_indices])

        try:
            start = 0
            losses = []
            for ibatch in range(num_batches):
                LOG.debug('Batch %d/%d', ibatch, num_batches)
                end = start + batch_size
                u_batch, v_batch = samples_u[start:end], samples_v[start:end]
                loss = train_step(model, u_batch, v_batch, optimizer, callback)
                start = end
                losses.append(loss)
                callback.train_eval(model, iepoch, ibatch, records)

            epoch_loss = np.mean(losses)
            if np.all(np.array(epoch_losses) > epoch_loss):
                if best_model_snapshot is not None:
                    best_model_snapshot.close()
                best_model_snapshot = h5py.File.in_memory()
                model.save(best_model_snapshot)

            epoch_losses.append(epoch_loss)

            test_loss = loss_fn(model, test_u, test_v)
            if callback.test(model, test_u, test_v, test_loss, iepoch, records):
                break

        except KeyboardInterrupt:
            LOG.info('Training interrupted by SIGINT')
            break

    if best_model_snapshot is None:
        best_model = None
    else:
        best_model = ConditionalRBM.load(best_model_snapshot)
        best_model_snapshot.close()

    return best_model, records
