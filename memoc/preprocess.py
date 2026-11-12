"""Volume preprocessing for ME-MoC: axis order, slice crop, width padding,
intensity normalisation and real/imaginary channel packing."""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def swap_to_nxye(volume: np.ndarray) -> np.ndarray:
    """Convert (H, W, S, E) -> (S, H, W, E)."""
    return np.swapaxes(np.swapaxes(volume, 0, 2), 1, 2)


def swap_to_hwse(nxye: np.ndarray) -> np.ndarray:
    """Convert (S, H, W, E) -> (H, W, S, E)."""
    return np.swapaxes(np.swapaxes(nxye, 1, 2), 0, 2)


def resolve_slice_bounds(num_slice: int, slice_rng) -> Tuple[int, int]:
    """Resolve zero-based, end-exclusive crop bounds without applying the crop.

    ``slice_rng`` is either None (all slices), ``[start, end]``, or a dict
    keyed by the slice count, e.g. ``{"96": [20, 82]}``.
    """
    if slice_rng is None:
        return 0, int(num_slice)
    if isinstance(slice_rng, (list, tuple)) and len(slice_rng) == 2:
        start, end = int(slice_rng[0]), int(slice_rng[1])
        return max(0, start), min(int(num_slice), end)

    if isinstance(slice_rng, dict):
        key_str = str(int(num_slice))
        key_int = int(num_slice)
        if key_str in slice_rng:
            start, end = slice_rng[key_str]
            return max(0, int(start)), min(int(num_slice), int(end))
        if key_int in slice_rng:
            start, end = slice_rng[key_int]
            return max(0, int(start)), min(int(num_slice), int(end))
        raise KeyError(f"slice_rng missing key for num_slice={num_slice}. Available: {sorted(slice_rng.keys())}")

    raise TypeError(f"Unsupported slice_rng type: {type(slice_rng)}")


def crop_slices(nxye: np.ndarray, slice_rng) -> np.ndarray:
    """Crop the slice dimension (axis 0) with ``resolve_slice_bounds``."""
    if slice_rng is None:
        return nxye
    start, end = resolve_slice_bounds(int(nxye.shape[0]), slice_rng)
    return nxye[start:end]


def pad_width_190_to_192(arr: np.ndarray) -> np.ndarray:
    """Force the width dimension (axis 2) to 192.

    - width == 192: unchanged
    - width < 192: right-pad by repeating the last column
    - width > 192: right-crop to 192
    """
    target_w = 192
    width = int(arr.shape[2])
    if width == target_w:
        return arr

    if width < target_w:
        pad_w = target_w - width
        pad_cfg = [(0, 0)] * arr.ndim
        pad_cfg[2] = (0, pad_w)
        return np.pad(arr, pad_cfg, mode="edge")

    return arr[:, :, :target_w, ...]


def norm_data_params(nxye: np.ndarray, method: str):
    """Return (offset, scale) used by ``norm_data``."""
    if method is None:
        return np.asarray(0.0, dtype=nxye.dtype), np.asarray(1.0, dtype=np.float32)
    method = str(method).lower()
    if method in ("none", "null"):
        return np.asarray(0.0, dtype=nxye.dtype), np.asarray(1.0, dtype=np.float32)
    if method == "maxmin":
        vmin = np.amin(nxye)
        vmax = np.amax(nxye)
        return vmin, vmax - vmin
    if method == "max":
        ref = np.amax(np.abs(nxye), axis=(1, 2), keepdims=True)
        return np.asarray(0.0, dtype=nxye.dtype), ref
    if method == "mean":
        ref = np.mean(np.abs(nxye), axis=(1, 2), keepdims=True)
        return np.asarray(0.0, dtype=nxye.dtype), ref
    if method == "midcube":
        # Mean magnitude of the middle slice over all echoes (paper default).
        num_slice = nxye.shape[0]
        ref = np.mean(np.abs(nxye[num_slice // 2, ...]))
        return np.asarray(0.0, dtype=nxye.dtype), ref

    raise ValueError(f"Unsupported norm method: {method}")


def norm_data(nxye: np.ndarray, method: str) -> np.ndarray:
    """Normalize (S, H, W, ...) volumes. Methods: none | maxmin | max | mean | midcube."""
    offset, scale = norm_data_params(nxye, method)
    return (nxye - offset) / (scale + 1e-12)


def denorm_data(nxye: np.ndarray, offset, scale) -> np.ndarray:
    """Invert ``norm_data`` using parameters returned by ``norm_data_params``."""
    return nxye * (scale + 1e-12) + offset


def make_channels(data: np.ndarray, output_type: str, new_dim: bool = False) -> np.ndarray:
    """Pack complex data into real channels.

    - complex: real and imaginary parts on the last axis
      (with new_dim=True a new trailing axis of size 2 is created)
    - magnitude / real / imag: the corresponding real array
    """
    output_type = str(output_type).lower()
    if output_type == "complex":
        if new_dim:
            return np.concatenate((data.real[..., None], data.imag[..., None]), axis=-1)
        return np.concatenate((data.real, data.imag), axis=-1)
    if output_type == "magnitude":
        return np.abs(data)
    if output_type == "real":
        return data.real
    if output_type == "imag":
        return data.imag
    raise ValueError(f"Unsupported output_type: {output_type}")
