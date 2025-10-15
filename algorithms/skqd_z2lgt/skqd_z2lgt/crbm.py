"""Conditional restricted Boltzmann machine."""
from functools import partial
import logging
from typing import Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax import nnx

LOG = logging.getLogger(__name__)


class ConditionalRBM(nnx.Module):
    """Conditional restricted Boltzmann machine."""
    def __init__(self, num_u: int, num_v: int, num_h: int, *, rngs: nnx.Rngs):
        init = nnx.initializers.normal(0.01)
        self.weights_vu = nnx.Param(init(rngs.params(), (num_v, num_u), jnp.float32))
        self.weights_hu = nnx.Param(init(rngs.params(), (num_h, num_u), jnp.float32))
        self.weights_hv = nnx.Param(init(rngs.params(), (num_h, num_v), jnp.float32))
        self.bias_v = nnx.Param(init(rngs.params(), (num_v,), jnp.float32))
        self.bias_h = nnx.Param(init(rngs.params(), (num_h,), jnp.float32))
        self.rngs = rngs
        self.therm_steps = 100
        self.vhat_size = 100

    def save(self, file: str | h5py.Group):
        params_state, rngs_state = map(nnx.pure, nnx.state(self, nnx.Param, ...))
        if isinstance(file, str):
            out = h5py.File(file, 'w')
        else:
            out = file

        params = out.create_group('params')
        for key, value in params_state.items():
            params.create_dataset(key, data=value)

        rngs = out.create_group('rngs')
        for key, state in rngs_state['rngs'].items():
            group = rngs.create_group(key)
            group.create_dataset('count', data=state['count'])
            group.create_dataset('key', data=jax.random.key_data(state['key']))

        out.create_dataset('therm_steps', data=self.therm_steps)
        out.create_dataset('vhat_size', data=self.vhat_size)

        if isinstance(file, str):
            out.close()

    @staticmethod
    def load(file: str | h5py.Group) -> 'ConditionalRBM':
        if isinstance(file, str):
            source = h5py.File(file)
        else:
            source = file

        params = {key: jnp.array(data[()]) for key, data in source['params'].items()}
        rngs_state = {}
        for key, state in source['rngs'].items():
            rngs_state[key] = {}
            try:
                rngs_state[key]['count'] = jnp.array(state['count'][()])
            except KeyError:
                LOG.error('Failed to load rngs/%s/count', key)
                rngs_state[key]['count'] = jnp.array(0, dtype=np.uint32)
            try:
                rngs_state[key]['key'] = jax.random.wrap_key_data(state['key'][()])
            except KeyError:
                LOG.error('Failed to load rngs/%s/key', key)
                rngs_state[key]['key'] = jnp.array([0, 0], dtype=np.uint32)
        therm_steps = source['therm_steps'][()]
        vhat_size = source['vhat_size'][()]

        if isinstance(file, str):
            source.close()

        rngs = nnx.Rngs(**{key: state['key'] for key, state in rngs_state.items()})
        for key, state in rngs_state.items():
            getattr(rngs, key).count.value = state['count']

        model = ConditionalRBM(
            params['weights_vu'].shape[1],
            params['bias_v'].shape[0],
            params['bias_h'].shape[0],
            rngs=rngs
        )
        for key, value in params.items():
            getattr(model, key).value = value

        model.therm_steps = therm_steps
        model.vhat_size = vhat_size

        return model

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

    @nnx.jit
    def conditional_logz(self, u_state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Log partition function given a u state. Intractable for num_v >~ 25.

        Since the raw sum of exp(-F) can be numerically unmanageable, we subtract the mean of F per
        sample before taking the exponential. This normalization factor will then be needed to
        compute the conditional probability and is therefore given as the second return value of
        this function.
        """
        num_v = self.bias_v.shape[0]
        all_v = ((jnp.arange(2 ** num_v)[:, None] >> jnp.arange(num_v)[None, :])
                 % 2).astype(np.uint8)
        all_f = self.vfree_energy(u_state, all_v)
        norm = jnp.mean(all_f, axis=0)
        all_f -= norm[None, ...]
        return jnp.log(jnp.sum(jnp.exp(-all_f), axis=0)), norm

    @nnx.jit
    def conditional_nll(self, u_state: jax.Array, v_state: jax.Array) -> jax.Array:
        """Probability of the v state given u state. Intractable for num_v >~ 25."""
        logz, norm = self.conditional_logz(u_state)
        return self.free_energy(u_state, v_state) - norm + logz

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
        batch_size = u_state.size // u_state.shape[-1]
        num_v = batch_size * self.bias_v.shape[0]
        num_h = batch_size * self.bias_h.shape[0]
        uniform_size = num_v + (num_v + num_h) * np.prod(size).astype(int)
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
        batch_size = u_state.size // u_state.shape[-1]
        num_v = batch_size * self.bias_v.shape[0]
        num_h = batch_size * self.bias_h.shape[0]

        if size is None:
            size = 1
            final_state_only = True
        if not isinstance(size, tuple):
            size = (int(size),)
        flat_size = np.prod(size).astype(int)

        def loop_body_generate(istep, val):
            module, u_state, v_state, uniform = val
            start = (num_h + num_v) * istep
            unif = jax.lax.dynamic_slice(uniform, [start], [num_h + num_v])
            ph = module.h_activation(u_state, v_state)
            h_state = (unif[:num_h].reshape(ph.shape) < ph).astype(np.uint8)
            pv = module.v_activation(u_state, h_state)
            v_state = (unif[num_h:].reshape(pv.shape) < pv).astype(np.uint8)
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

    @nnx.jit
    def meanloss(
        self,
        u_state: jax.Array,
        v_state: jax.Array,
        vg_states: jax.Array
    ) -> jax.Array:
        return (self.free_energy(u_state, v_state)
                - jnp.mean(self.vfree_energy(u_state, vg_states), axis=0))

    @partial(nnx.jit, static_argnames=['size'])
    def sample(
        self,
        u_state: jax.Array,
        size: Optional[int | tuple[int, ...]] = None
    ):
        batch_size = u_state.size // u_state.shape[-1]
        num_v = batch_size * self.bias_v.shape[0]
        num_h = batch_size * self.bias_h.shape[0]
        if size is None:
            gen_size = 1
        else:
            gen_size = np.prod(size).astype(int)
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
