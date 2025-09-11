"""Conditional restricted Boltzmann machine."""
from collections.abc import Callable
from functools import partial
from itertools import product
import logging
from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

LOG = logging.getLogger(__name__)


class ConditionalRBM(nnx.Module):
    """Conditional restricted Boltzmann machine."""
    def __init__(self, num_u: int, num_v: int, num_h: int, *, rngs: nnx.Rngs):
        weights_init = nnx.initializers.lecun_normal()
        bias_init = nnx.initializers.normal()
        self.weights_vu = nnx.Param(weights_init(rngs.params(), (num_v, num_u), jnp.float32))
        self.weights_hu = nnx.Param(weights_init(rngs.params(), (num_h, num_u), jnp.float32))
        self.weights_hv = nnx.Param(weights_init(rngs.params(), (num_h, num_v), jnp.float32))
        self.bias_v = nnx.Param(bias_init(rngs.params(), (num_v,), jnp.float32))
        self.bias_h = nnx.Param(bias_init(rngs.params(), (num_h,), jnp.float32))
        self.rngs = rngs
        self.therm_steps = 100
        self.vhat_size = 100

    @nnx.jit
    def energy(self, u_state: jax.Array, v_state: jax.Array, h_state: jax.Array) -> jax.Array:
        """Energy function."""
        e_val = -(h_state[:, None, :] @ self.weights_hv @ v_state[:, :, None]
                  + v_state[:, None, :] @ self.weights_vu @ u_state[:, :, None]
                  + h_state[:, None, :] @ self.weights_hu @ u_state[:, :, None])
        e_val = jnp.squeeze(e_val, axis=(1, 2))
        e_val -= v_state @ self.bias_v + h_state @ self.bias_h
        return e_val

    @nnx.jit
    def free_energy(self, u_state: jax.Array, v_state: jax.Array) -> jax.Array:
        """Free energy -log(sum_h(exp(-E(v,h,u))))."""
        f_val = -jnp.sum(v_state * (self.bias_v + u_state @ self.weights_vu.T), axis=-1)
        f_val -= jnp.sum(
            jnp.log(1. + jnp.exp(self.bias_h + v_state @ self.weights_hv.T
                                 + u_state @ self.weights_hu.T)),
            axis=-1
        )
        return f_val

    vfree_energy = nnx.jit(nnx.vmap(free_energy, in_axes=(None, None, 0)))
    bvfree_energy = nnx.jit(nnx.vmap(vfree_energy, in_axes=(None, 0, None)))

    @nnx.jit
    def conditional_partition_function(self, u_state: jax.Array) -> jax.Array:
        """Partition function given a u state. Intractable for num_v >~ 25."""
        num_v = self.bias_v.shape[0]
        all_v = ((jnp.arange(2 ** num_v)[:, None] >> jnp.arange(num_v)[None, :])
                 % 2).astype(np.uint8)
        return jnp.sum(jnp.exp(-self.bvfree_energy(u_state, all_v)), axis=1)

    @nnx.jit
    def conditional_probability(self, u_state: jax.Array, v_state: jax.Array) -> jax.Array:
        """Probability of the v state given u state. Intractable for num_v >~ 25."""
        partfun = self.conditional_partition_function(u_state)
        return jnp.exp(-self.free_energy(u_state, v_state)) / partfun

    @nnx.jit
    def h_activation(self, u_state: jax.Array, v_state: jax.Array) -> jax.Array:
        delta_e = v_state @ self.weights_hv.T + u_state @ self.weights_hu.T + self.bias_h
        return nnx.sigmoid(delta_e)

    @nnx.jit
    def v_activation(self, u_state: jax.Array, h_state: jax.Array) -> jax.Array:
        delta_e = h_state @ self.weights_hv + u_state @ self.weights_vu.T + self.bias_v
        return nnx.sigmoid(delta_e)

    @partial(nnx.jit, static_argnames=['size', 'final_state_only'])
    def gibbs_sample(
        self,
        u_state: jax.Array,
        v_state: jax.Array,
        size: int | tuple[int, ...] = 1,
        final_state_only: bool = False
    ) -> jax.Array:
        """MCMC sample generation."""
        batch_size = np.prod(u_state.shape[:-1])
        num_v = batch_size * self.bias_v.shape[0]
        num_h = batch_size * self.bias_h.shape[0]
        uniform_size = num_v + (num_v + num_h) * np.prod(size)
        uniform = jax.random.uniform(self.rngs.sample(), (uniform_size,))
        return self._gibbs_sample(u_state, v_state, uniform, size=size,
                                  final_state_only=final_state_only)

    @partial(nnx.jit, static_argnames=['size', 'final_state_only'])
    def _gibbs_sample(
        self,
        u_state: jax.Array,
        v_state: jax.Array,
        uniform: jax.Array,
        size: Optional[int | tuple[int, ...]] = None,
        final_state_only: bool = False
    ) -> jax.Array:
        """MCMC sample generation."""
        batch_size = np.prod(u_state.shape[:-1])
        num_v = batch_size * self.bias_v.shape[0]
        num_h = batch_size * self.bias_h.shape[0]

        def generate_v_state(module, u_state, v_state, uniform):
            ph = module.h_activation(u_state, v_state)
            h_state = (uniform[:num_h].reshape(ph.shape) < ph).astype(np.uint8)
            pv = module.v_activation(u_state, h_state)
            return (uniform[num_h:].reshape(pv.shape) < pv).astype(np.uint8)

        if size is None:
            size = 1
            final_state_only = True
        if not isinstance(size, tuple):
            size = (int(size),)
        flat_size = np.prod(size)

        def loop_body_generate(istep, val):
            module, u_state, v_state, uniform = val
            start = (num_h + num_v) * istep
            unif = jax.lax.dynamic_slice(uniform, [start], [num_h + num_v])
            v_state = generate_v_state(module, u_state, v_state, unif)
            return module, u_state, v_state, uniform

        if final_state_only:
            loop_body = loop_body_generate
            init_val = (self, u_state, v_state, uniform)
        else:
            def loop_body(istep, val):
                module, u_state, v_state, uniform = loop_body_generate(istep, val[:-1])
                out = val[-1]
                return module, u_state, v_state, uniform, out.at[istep].set(v_state)

            out = jnp.empty((flat_size,) + v_state.shape, dtype=np.uint8)
            init_val = (self, u_state, v_state, uniform, out)

        final_val = nnx.fori_loop(
            0, flat_size,
            loop_body,
            init_val
        )
        if final_state_only:
            return final_val[2]
        return final_val[-1].reshape(size + v_state.shape)

    @nnx.jit
    def percloss_states(self, u_state: jax.Array):
        v_states = self.sample(u_state, self.vhat_size)
        free_energies = self.vfree_energy(u_state, v_states)
        min_indices = jnp.argmin(free_energies, axis=0)
        return v_states[min_indices, jnp.arange(u_state.shape[0])]

    @nnx.jit
    def percloss(
        self,
        u_state: jax.Array,
        v_state: jax.Array,
        vhat_state: jax.Array
    ) -> jax.Array:
        return self.free_energy(u_state, v_state) - self.free_energy(u_state, vhat_state)

    @partial(nnx.jit, static_argnames=['size'])
    def sample(
        self,
        u_state: jax.Array,
        size: Optional[int | tuple[int, ...]] = None
    ):
        batch_size = np.prod(u_state.shape[:-1])
        num_v = batch_size * self.bias_v.shape[0]
        num_h = batch_size * self.bias_h.shape[0]
        if size is None:
            gen_size = 1
        else:
            gen_size = np.prod(size)
        uniform_size = num_v + (num_v + num_h) * (self.therm_steps + gen_size)
        uniform = jax.random.uniform(self.rngs.sample(), (uniform_size,))

        pv = nnx.sigmoid(u_state @ self.weights_vu.T + self.bias_v)
        v_state = (uniform[:num_v].reshape(pv.shape) < pv).astype(np.uint8)
        start = num_v
        end = num_v + (num_v + num_h) * self.therm_steps
        v_state = self._gibbs_sample(u_state, v_state, uniform[start:end], size=self.therm_steps,
                                     final_state_only=True)
        start = end
        return self._gibbs_sample(u_state, v_state, uniform[start:], size=size)


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


@nnx.jit
def loss_and_free_energy(model, u_batch, v_batch):
    vhat_batch = model.percloss_states(u_batch)
    loss = loss_fn(model, u_batch, v_batch, vhat_batch)
    free_energy = jnp.mean(model.free_energy(u_batch, v_batch))
    return loss, free_energy


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

    def test(
        self,
        model: ConditionalRBM,
        test_u: jax.Array,
        test_v: jax.Array,
        iepoch: int,
        records: dict[str, Any]
    ):
        loss, free_energy = loss_and_free_energy(model, test_u, test_v)
        self.metrics.update(loss=loss, free_energy=free_energy)
        for metric, value in self.metrics.compute().items():
            records[f'test_{metric}'].append(float(value))
        self.metrics.reset()
