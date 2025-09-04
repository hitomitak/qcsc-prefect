"""Conditional restricted Boltzmann machine."""
from collections.abc import Callable
from functools import partial
import logging
from typing import Optional
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
        f_val = -jnp.sum(v_state * (self.bias_v[None, :] + u_state @ self.weights_vu.T), axis=1)
        f_val -= jnp.sum(
            jnp.log(1. + jnp.exp(self.bias_h[None, :]
                                 + v_state @ self.weights_hv.T
                                 + u_state @ self.weights_hu.T)),
            axis=1
        )
        return f_val

    vfree_energy = nnx.jit(nnx.vmap(free_energy, in_axes=(None, None, 0), out_axes=0))

    @nnx.jit
    def h_activation(self, u_state: jax.Array, v_state: jax.Array) -> jax.Array:
        delta_e = v_state @ self.weights_hv.T + u_state @ self.weights_hu.T + self.bias_h
        return nnx.sigmoid(delta_e)

    @nnx.jit
    def v_activation(self, u_state: jax.Array, h_state: jax.Array) -> jax.Array:
        delta_e = h_state @ self.weights_hv + u_state @ self.weights_vu.T + self.bias_v
        return nnx.sigmoid(delta_e)

    @partial(nnx.jit, static_argnames=['size'])
    def gibbs_sample(
        self,
        u_state: jax.Array,
        v_state: jax.Array,
        size: Optional[int | tuple[int, ...]] = None
    ) -> jax.Array:
        def generate_v_state(module, u_state, v_state):
            ph = module.h_activation(u_state, v_state)
            h_state = jax.random.binomial(module.rngs.sample(), 1, ph).astype(np.uint8)
            pv = module.v_activation(u_state, h_state)
            return jax.random.binomial(module.rngs.sample(), 1, pv).astype(np.uint8)

        if size is None:
            return generate_v_state(self, u_state, v_state)

        if isinstance(size, int):
            size = (size,)
        flat_size = np.prod(size)

        def fill_samples(isample, val):
            module, u_state, v_state, out = val
            v_state = generate_v_state(module, u_state, v_state)
            return module, u_state, v_state, out.at[isample].set(v_state)

        out = jnp.empty((flat_size,) + v_state.shape, dtype=np.uint8)
        out = nnx.fori_loop(
            0, flat_size,
            fill_samples,
            (self, u_state, v_state, out)
        )[-1]
        return out.reshape(size + v_state.shape)

    @partial(nnx.jit, static_argnames=['num_gen'])
    def percloss_states(self, u_state: jax.Array, num_gen: int) -> jax.Array:
        v_states = self.sample(u_state, size=num_gen)
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

    @partial(nnx.jit, static_argnames=['size', 'therm_steps'])
    def sample(
        self,
        u_state: jax.Array,
        size: Optional[int | tuple[int, ...]] = None,
        therm_steps: int = 100
    ):
        pv = nnx.sigmoid(u_state @ self.weights_vu.T + self.bias_v)
        v_state = jax.random.binomial(self.rngs.sample(), 1, pv).astype(np.uint8)
        v_state = self.gibbs_sample(u_state, v_state, size=therm_steps)[-1]
        return self.gibbs_sample(u_state, v_state, size=size)


def train_crbm(
    model: ConditionalRBM,
    train_dataset: np.ndarray,
    test_dataset: np.ndarray,
    batch_size: int,
    num_epochs: int,
    eval_every: int = 10,
    optax_fn: Optional[Callable] = None,
    seed: int = 0,
    cdpl_num_gen: int = 100
):
    def loss_fn(model: ConditionalRBM, u_state, v_state, vhat_state):
        return jnp.mean(model.percloss(u_state, v_state, vhat_state))

    grad_fn = nnx.value_and_grad(loss_fn)

    @nnx.jit
    def train_step(model, optimizer, metrics, u_batch, v_batch):
        vhat_batch = model.percloss_states(u_batch, cdpl_num_gen)
        loss, grads = grad_fn(model, u_batch, v_batch, vhat_batch)
        free_energy = jnp.mean(model.free_energy(u_batch, v_batch))
        metrics.update(loss=loss, free_energy=free_energy)
        optimizer.update(model, grads)

    @nnx.jit
    def eval_step(model, metrics, u_batch, v_batch):
        vhat_batch = model.percloss_states(u_batch, cdpl_num_gen)
        loss = loss_fn(model, u_batch, v_batch, vhat_batch)
        free_energy = jnp.mean(model.free_energy(u_batch, v_batch))
        metrics.update(loss=loss, free_energy=free_energy)

    optax_fn = optax_fn or optax.adamw(learning_rate=0.005)
    optimizer = nnx.Optimizer(model, optax_fn, wrt=nnx.Param)

    metrics = nnx.metrics.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        free_energy=nnx.metrics.Average('free_energy')
    )

    rng = np.random.default_rng(seed)
    num_batches = train_dataset.shape[0] // batch_size
    num_u = model.weights_hu.shape[1]

    test_u = jax.device_put(test_dataset[:, :num_u])
    test_v = jax.device_put(test_dataset[:, num_u:])

    metrics_history = {
        'train_loss': [],
        'train_free_energy': [],
        'test_loss': [],
        'test_free_energy': []
    }

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
            train_step(model, optimizer, metrics, u_batch, v_batch)

            if ibatch % eval_every == 0 or ibatch == num_batches - 1:
                for metric, value in metrics.compute().items():
                    metrics_history[f'train_{metric}'].append(value)
                metrics.reset()

        eval_step(model, metrics, test_u, test_v)
        for metric, value in metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)
        metrics.reset()

    return metrics_history
