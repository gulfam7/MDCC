"""tf.data pipeline that streams (x, y) slice pairs from ME-MoC HDF5 caches."""
from __future__ import annotations

from dataclasses import dataclass
from threading import local
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import tensorflow as tf


_THREAD_LOCAL = local()


@dataclass(frozen=True)
class Record:
    cache_file: str
    slice_index: int


def _get_handle_cache():
    cache = getattr(_THREAD_LOCAL, "h5_handles", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.h5_handles = cache
    return cache


def _open_cached_h5(path: str) -> h5py.File:
    cache = _get_handle_cache()
    handle = cache.get(path)
    if handle is not None and getattr(handle, "id", None) and handle.id.valid:
        return handle
    handle = h5py.File(path, "r")
    cache[path] = handle
    return handle


def _list_records(cache_files: Sequence[str]) -> List[Record]:
    records: List[Record] = []
    for cache_file in cache_files:
        with h5py.File(cache_file, "r") as f:
            n = int(f["x"].shape[0])
        records.extend(Record(cache_file=cache_file, slice_index=i) for i in range(n))
    return records


def count_records(cache_files: Sequence[str]) -> int:
    total = 0
    for cache_file in cache_files:
        with h5py.File(cache_file, "r") as f:
            total += int(f["x"].shape[0])
    return total


def infer_sample_shape(cache_files: Sequence[str]) -> Tuple[int, ...]:
    if not cache_files:
        raise ValueError("infer_sample_shape requires at least one cache file.")
    with h5py.File(cache_files[0], "r") as f:
        x = f["x"]
        if x.shape[0] == 0:
            raise ValueError(f"Cache file has no samples: {cache_files[0]}")
        return tuple(x.shape[1:])


def _decode_cache_path(cache_file) -> str:
    if hasattr(cache_file, "numpy"):
        cache_file = cache_file.numpy()
    if isinstance(cache_file, (bytes, bytearray, np.bytes_)):
        return cache_file.decode("utf-8")
    return str(cache_file)


def _load_record_xy_py(cache_file: bytes, slice_index: np.ndarray):
    cache_path = _decode_cache_path(cache_file)
    idx = int(slice_index)
    f = _open_cached_h5(cache_path)
    x = f["x"][idx]
    y = f["y"][idx]
    return x.astype(np.float32), y.astype(np.float32)


def make_tf_dataset(
    cache_files: Sequence[str],
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Return a batched dataset of (x, y) slices, each shaped (H, W, E, 2)."""
    records = _list_records(cache_files)
    if not records:
        raise ValueError("No records found (empty cache_files or empty shard datasets).")

    sample_shape = infer_sample_shape(cache_files)
    target_h = int(sample_shape[0])
    target_w = int(sample_shape[1])

    def _fit_hw(t: tf.Tensor) -> tf.Tensor:
        t = t[:target_h, :target_w, ...]
        h = tf.shape(t)[0]
        w = tf.shape(t)[1]
        pad_h = tf.maximum(0, target_h - h)
        pad_w = tf.maximum(0, target_w - w)
        return tf.pad(t, [[0, pad_h], [0, pad_w], [0, 0], [0, 0]])

    cache_file_tensor = tf.constant([r.cache_file for r in records])
    slice_tensor = tf.constant([r.slice_index for r in records], dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((cache_file_tensor, slice_tensor))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(records), 4096), seed=seed, reshuffle_each_iteration=True)

    def _map_fn_xy(path, idx):
        x, y = tf.py_function(func=_load_record_xy_py, inp=[path, idx], Tout=[tf.float32, tf.float32])
        x = _fit_hw(x)
        y = _fit_hw(y)
        x.set_shape([target_h, target_w, sample_shape[2], 2])
        y.set_shape([target_h, target_w, sample_shape[2], 2])
        return x, y

    ds = ds.map(_map_fn_xy, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
