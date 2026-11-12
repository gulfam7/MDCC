"""Volume preprocessing for ME-MoD: axis order, slice crop, width padding,
intensity normalisation and real/imaginary channel packing."""
from __future__ import annotations

import os

import numpy as np


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def swap_to_nxye(volume: np.ndarray) -> np.ndarray:
    """Convert (H, W, S, E) -> (S, H, W, E)."""
    return np.swapaxes(np.swapaxes(volume, 0, 2), 1, 2)


def crop_slices(nxye: np.ndarray, slice_rng) -> np.ndarray:
    """Crop the slice dimension (axis 0).

    ``slice_rng`` is None (all slices), zero-based end-exclusive
    ``[start, end]``, or a dict keyed by the slice count, e.g. ``{"96": [29, 75]}``.
    """
    if slice_rng is None:
        return nxye
    if isinstance(slice_rng, (list, tuple)) and len(slice_rng) == 2:
        start, end = int(slice_rng[0]), int(slice_rng[1])
        return nxye[start:end]

    if isinstance(slice_rng, dict):
        num_slice = int(nxye.shape[0])
        key_str = str(num_slice)
        key_int = num_slice
        if key_str in slice_rng:
            start, end = slice_rng[key_str]
            return nxye[int(start) : int(end)]
        if key_int in slice_rng:
            start, end = slice_rng[key_int]
            return nxye[int(start) : int(end)]
        raise KeyError(f"slice_rng missing key for num_slice={num_slice}. Available: {sorted(slice_rng.keys())}")

    raise TypeError(f"Unsupported slice_rng type: {type(slice_rng)}")


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


def norm_data(nxye: np.ndarray, method: str) -> np.ndarray:
    """Normalize (S, H, W, ...) volumes.

    Methods: none | maxmin | max | mean | midcube | echo1mean. ME-MoD uses
    ``echo1mean``: division by the mean non-zero echo-1 magnitude of the volume.
    """
    if method is None:
        return nxye
    method = str(method).lower()
    if method in ("none", "null"):
        return nxye
    if method == "maxmin":
        vmin = np.amin(nxye)
        vmax = np.amax(nxye)
        return (nxye - vmin) / (vmax - vmin + 1e-12)
    if method == "max":
        ref = np.amax(np.abs(nxye), axis=(1, 2), keepdims=True)
        return nxye / (ref + 1e-12)
    if method == "mean":
        ref = np.mean(np.abs(nxye), axis=(1, 2), keepdims=True)
        return nxye / (ref + 1e-12)
    if method == "midcube":
        num_slice = nxye.shape[0]
        ref = np.mean(np.abs(nxye[num_slice // 2, ...]))
        return nxye / (ref + 1e-12)
    if method == "echo1mean":
        echo1 = nxye[..., 0]
        mag = np.abs(echo1)
        nonzero = mag > 0
        if np.any(nonzero):
            ref = np.mean(mag[nonzero])
        else:
            ref = np.mean(mag)
        return nxye / (ref + 1e-12)

    raise ValueError(f"Unsupported norm method: {method}")


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
