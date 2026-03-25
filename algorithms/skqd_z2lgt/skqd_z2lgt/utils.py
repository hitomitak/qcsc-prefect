"""Utility functions."""
from typing import Any, Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


def read_bits(
    dataset: np.ndarray | h5py.Dataset,
    num_bits: Optional[int] = None,
    align: str = 'left',
    offset: int = 0
) -> np.ndarray:
    bits = np.unpackbits(dataset, axis=-1)
    num_bits = num_bits or dataset.attrs['num_bits']
    if align == 'left':
        return bits[..., offset:num_bits + offset]
    return bits[..., -(offset + num_bits):bits.shape[-1] - offset]


def save_bits(group: h5py.Group, name: str, bits: np.ndarray) -> h5py.Dataset:
    dataset = group.create_dataset(name, data=np.packbits(bits, axis=-1))
    dataset.attrs['num_bits'] = bits.shape[-1]
    return dataset


def shard_array_1d(
    array: jax.Array,
    fill_value: Optional[Any] = None,
    device_ids: Optional[list[int]] = None
) -> jax.Array:
    """Shard the given array along dimension 0."""
    length = array.shape[0]
    if device_ids:
        num_dev = len(device_ids)
        devices = [jax.devices()[did] for did in device_ids]
    else:
        num_dev = jax.device_count()
        devices = None
    terms_per_device = int(np.ceil(length / num_dev).astype(int))
    residual = num_dev * terms_per_device - length
    if residual > 0:
        if fill_value is None:
            padding = jnp.zeros((residual,) + array.shape[1:], dtype=array.dtype)
        else:
            padding = jnp.full((residual,) + array.shape[1:], fill_value, dtype=array.dtype)
        array = jnp.concatenate([array, padding], axis=0)
    mesh = jax.make_mesh((num_dev,), ('device',), devices=devices)
    return jax.device_put(array, NamedSharding(mesh, PartitionSpec('device')))
