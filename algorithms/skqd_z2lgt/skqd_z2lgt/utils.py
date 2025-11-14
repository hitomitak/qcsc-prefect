"""Utility functions."""
from typing import Any, Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


def read_bits(dataset: np.ndarray | h5py.Dataset, num_bits: Optional[int] = None, align='left'):
    bits = np.unpackbits(dataset, axis=-1)
    num_bits = num_bits or dataset.attrs['num_bits']
    if align == 'left':
        return bits[..., :num_bits]
    return bits[..., -num_bits:]


def shard_array_1d(array: jax.Array, fill_value: Optional[Any] = None) -> jax.Array:
    """Shard the given array along dimension 0."""
    length = array.shape[0]
    num_dev = jax.device_count()
    terms_per_device = int(np.ceil(length / num_dev).astype(int))
    residual = num_dev * terms_per_device - length
    if residual > 0:
        if fill_value is None:
            padding = jnp.zeros((residual,) + array.shape[1:], dtype=array.dtype)
        else:
            padding = jnp.full((residual,) + array.shape[1:], fill_value, dtype=array.dtype)
        array = jnp.concatenate([array, padding], axis=0)
    mesh = jax.make_mesh((num_dev,), ('device',))
    return jax.device_put(array, NamedSharding(mesh, PartitionSpec('device')))
