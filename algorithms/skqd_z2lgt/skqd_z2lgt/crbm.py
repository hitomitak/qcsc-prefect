"""Conditional restricted Boltzmann machine."""
from functools import partial
import logging
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

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
