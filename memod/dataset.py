"""tf.data pipeline that yields padded bags of slices from ME-MoD caches."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tensorflow as tf


def infer_sample_shape(cache_files: Sequence[str]) -> Tuple[int, int, int]:
    if not cache_files:
        raise ValueError("infer_sample_shape requires at least one cache file")
    with h5py.File(cache_files[0], "r") as f:
        return tuple(int(v) for v in f["x"].shape[1:])


def count_subject_samples(cache_files: Sequence[str]) -> int:
    return int(len(list(cache_files)))


def _decode_path(path_t) -> str:
    if hasattr(path_t, "numpy"):
        path_t = path_t.numpy()
    if isinstance(path_t, (bytes, bytearray, np.bytes_)):
        return path_t.decode("utf-8")
    return str(path_t)


def _load_subject_py(path_t):
    cache_path = _decode_path(path_t)
    with h5py.File(cache_path, "r") as f:
        x = np.asarray(f["x"][:], dtype=np.float32)
        y_subject = float(np.asarray(f["y_subject"][:]).reshape(-1)[0])
        y_slice = np.asarray(f["y_slice"][:], dtype=np.float32)
        y_slice_available = np.asarray(f["y_slice_available"][:], dtype=np.float32)
    return x, np.ones((x.shape[0],), dtype=np.float32), y_slice, y_slice_available, np.asarray(y_subject, dtype=np.float32)


def make_tf_subject_dataset(
    cache_files: Sequence[str],
    batch_size: int,
    shuffle: bool,
    seed: int,
    max_slices: Optional[int] = None,
    training: bool = True,
) -> tf.data.Dataset:
    """Return batches ``((x, mask, y_slice, y_slice_available), y_subject)``.

    Volumes with different slice counts are zero-padded to the longest bag in
    the batch; ``mask`` marks the real slices.
    """
    if not cache_files:
        raise ValueError("cache_files must not be empty")

    slice_shape = infer_sample_shape(cache_files)
    target_h, target_w, target_c = [int(v) for v in slice_shape]
    path_tensor = tf.constant([str(p) for p in cache_files])
    ds = tf.data.Dataset.from_tensor_slices(path_tensor)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(cache_files), 4096), seed=seed, reshuffle_each_iteration=True)

    def _load_subject(path):
        x, mask, y_slice, y_slice_available, y_subject = tf.py_function(
            func=_load_subject_py,
            inp=[path],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        )
        x.set_shape([None, None, None, None])
        mask.set_shape([None])
        y_slice.set_shape([None])
        y_slice_available.set_shape([None])
        y_subject.set_shape([])

        cur_h = tf.shape(x)[1]
        cur_w = tf.shape(x)[2]
        off_h = tf.maximum(0, (cur_h - target_h) // 2)
        off_w = tf.maximum(0, (cur_w - target_w) // 2)
        end_h = off_h + tf.minimum(cur_h, target_h)
        end_w = off_w + tf.minimum(cur_w, target_w)
        x = x[:, off_h:end_h, off_w:end_w, :]
        pad_h = tf.maximum(0, target_h - tf.shape(x)[1])
        pad_w = tf.maximum(0, target_w - tf.shape(x)[2])
        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        x = x[..., :target_c]
        pad_c = tf.maximum(0, target_c - tf.shape(x)[-1])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, pad_c]])
        x = x[:, :target_h, :target_w, :target_c]
        x.set_shape([None, target_h, target_w, target_c])

        if max_slices is not None:
            k = tf.constant(int(max_slices), dtype=tf.int32)
            s = tf.shape(x)[0]
            if training:
                perm = tf.random.shuffle(tf.range(s))[: tf.minimum(s, k)]
                x = tf.gather(x, perm)
                y_slice = tf.gather(y_slice, perm)
                y_slice_available = tf.gather(y_slice_available, perm)
            else:
                start = tf.maximum(0, (s - k) // 2)
                span = tf.minimum(s, k)
                x = x[start : start + span]
                y_slice = y_slice[start : start + span]
                y_slice_available = y_slice_available[start : start + span]
            mask = tf.ones((tf.shape(x)[0],), dtype=tf.float32)

        return (x, mask, y_slice, y_slice_available), y_subject

    ds = ds.map(_load_subject, num_parallel_calls=tf.data.AUTOTUNE)
    if max_slices is not None:
        pad_s = int(max_slices)
        padded_shapes = (([pad_s, target_h, target_w, target_c], [pad_s], [pad_s], [pad_s]), [])
    else:
        padded_shapes = (([None, target_h, target_w, target_c], [None], [None], [None]), [])
    padding_values = (
        (
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(-1.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
        ),
        tf.constant(0.0, dtype=tf.float32),
    )
    ds = ds.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_subject_class_weights(cache_files: Sequence[str]) -> Optional[dict]:
    """Balanced class weights {0: w0, 1: w1} from the volume labels (None if one class)."""
    ys: List[int] = []
    for path in cache_files:
        with h5py.File(path, "r") as f:
            ys.append(int(np.asarray(f["y_subject"][:]).reshape(-1)[0] > 0.5))
    if not ys:
        return None
    y = np.asarray(ys, dtype=np.int32)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    if n0 == 0 or n1 == 0:
        return None
    total = n0 + n1
    return {0: float(total / (2.0 * n0)), 1: float(total / (2.0 * n1))}
