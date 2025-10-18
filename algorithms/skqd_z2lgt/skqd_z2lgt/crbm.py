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
        # Generate uniform random values for all sampling uses in a single
        # call to jax.random.uniform(). Makes sampling faster but seems to cause GPU memory
        # leak.
        self.pregenerate = False

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
        init_v_state: jax.Array,
        size: Optional[int | tuple[int, ...]] = None,
        final_state_only: bool = False,
        uniform: Optional[jax.Array] = None
    ) -> jax.Array:
        """MCMC sample generation."""
        if size is None:
            size = 1
            final_state_only = True
        if not isinstance(size, tuple):
            size = (int(size),)
        flat_size = np.prod(size).astype(int)

        if uniform is not None:
            uniform = uniform.reshape((flat_size, -1))

        if final_state_only:
            if uniform is None:
                def loop_body(_, val):
                    module, _u_state, _v_state = val
                    _v_state = module._gibbs_sample_step(_u_state, _v_state)
                    return module, _u_state, _v_state

                init = (self, u_state, init_v_state)
            else:
                def loop_body(istep, val):
                    module, _uniform, _u_state, _v_state = val
                    _v_state = module._gibbs_sample_step(_u_state, _v_state, _uniform[istep])
                    return module, _uniform, _u_state, _v_state

                init = (self, uniform, u_state, init_v_state)
        else:
            out = jnp.empty((flat_size,) + init_v_state.shape, dtype=init_v_state.dtype)

            if uniform is None:
                def loop_body(istep, val):
                    module, _u_state, _v_state, _out = val
                    _v_state = module._gibbs_sample_step(_u_state, _v_state)
                    _out = _out.at[istep].set(_v_state)
                    return module, _u_state, _v_state, _out

                init = (self, u_state, init_v_state, out)
            else:
                def loop_body(istep, val):
                    module, _uniform, _u_state, _v_state, _out = val
                    _v_state = module._gibbs_sample_step(_u_state, _v_state, _uniform[istep])
                    _out = _out.at[istep].set(_v_state)
                    return module, _uniform, _u_state, _v_state, _out

                init = (self, uniform, u_state, init_v_state, out)

        return nnx.fori_loop(0, flat_size, loop_body, init)[-1]

    @nnx.jit
    def _gibbs_sample_step(self, u_state, v_state, uniform=None):
        if uniform is None:
            batch_size = u_state.size // u_state.shape[-1]
            num_h = batch_size * self.bias_h.shape[0]
            num_v = batch_size * self.bias_v.shape[0]
            uniform = jax.random.uniform(self.rngs.sample(), (num_v + num_h,))
        ph = self.h_activation(u_state, v_state)
        h_state = (uniform[:num_h] < ph.reshape(-1)).astype(np.uint8).reshape(ph.shape)
        pv = self.v_activation(u_state, h_state)
        v_state = (uniform[num_h:] < pv.reshape(-1)).astype(np.uint8).reshape(pv.shape)
        return v_state

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
    ) -> jax.Array:
        """Generate samples of v vectors.

        Returns the result of Gibbs sampling from a thermalized state. Thermalization is performed
        by `self.therm_steps` iterations of Gibbs sampling on the initial v vector drawn from
        v_activation(u, 0).

        Args:
            u_state: Condition bits. Shape [..., num_u].
            size: Number of samples to generate for each condition vector.

        Returns:
            Array of v vector samples with shape [*size, *u_state.shape[:-1], num_v].

        """
        pv = nnx.sigmoid(u_state @ self.weights_vu.T + self.bias_v)

        if self.pregenerate:
            batch_size = u_state.size // u_state.shape[-1]
            num_v = batch_size * self.bias_v.shape[0]
            num_h = batch_size * self.bias_h.shape[0]
            if size is None:
                size_flat = 1
            else:
                size_flat = np.prod(size).astype(int)
            init_stop = num_v
            therm_stop = init_stop + (num_v + num_h) * self.therm_steps
            uniform_size = therm_stop + (num_v + num_h) * size_flat
            uniform_all = jax.random.uniform(self.rngs.sample(), (uniform_size,))
            uniform = uniform_all[:init_stop].reshape(pv.shape)
            uniform_therm = uniform[init_stop:therm_stop]
            uniform_gen = uniform[therm_stop:]
        else:
            uniform = jax.random.uniform(self.rngs.sample(), pv.shape)
            uniform_therm = None
            uniform_gen = None

        v_state = (uniform < pv).astype(np.uint8)
        v_state = self.gibbs_sample(u_state, v_state, size=self.therm_steps, final_state_only=True,
                                    uniform=uniform_therm)
        return self.gibbs_sample(u_state, v_state, size=size, uniform=uniform_gen)
