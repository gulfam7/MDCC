"""Build HDF5 slice caches of paired (motion-corrupted, motion-free) volumes for ME-MoC.

Each configured motion label produces one shard per mode
(``<cache_root>/memoc_v1/<dataset>/<mode>/<signature>/<mode>__<label>.h5``).
A shard stores, per slice, the network input ``x`` (motion-corrupted) and
target ``y`` (motion-free), both shaped (H, W, E, 2), together with the slice
metadata needed to restore predictions into the full volume.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import h5py
import numpy as np

from .mat_io import load_mat_var
from .preprocess import (
    crop_slices,
    ensure_dir,
    make_channels,
    norm_data,
    pad_width_190_to_192,
    resolve_slice_bounds,
    swap_to_nxye,
)


CACHE_SCHEMA_VERSION = "memoc_v1"


@dataclass
class CachePaths:
    cache_root: str

    def mode_dir(self, dataset: str, mode: str, signature: str) -> str:
        return os.path.join(self.cache_root, CACHE_SCHEMA_VERSION, dataset, mode, signature)

    def shard_cache(self, dataset: str, mode: str, signature: str, label_suffix: str) -> str:
        return os.path.join(self.mode_dir(dataset, mode, signature), f"{mode}__{label_suffix}.h5")


class _ShardWriter:
    def __init__(
        self,
        path: str,
        dataset: str,
        mode: str,
        label: str,
        signature: str,
    ) -> None:
        ensure_dir(os.path.dirname(path))
        self.path = path
        self._f = h5py.File(path, "w")
        self._f.attrs.update(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "dataset": dataset,
                "mode": mode,
                "label": label,
                "config_signature": signature,
            }
        )

    def close(self) -> None:
        self._f.close()

    def _append_numeric(self, name: str, data: np.ndarray) -> None:
        data = np.asarray(data)
        if name not in self._f:
            self._f.create_dataset(
                name,
                data=data,
                maxshape=(None,) + data.shape[1:],
                compression="gzip",
                compression_opts=1,
                chunks=True,
            )
            return

        ds = self._f[name]
        old_n = int(ds.shape[0])
        new_n = old_n + int(data.shape[0])
        ds.resize((new_n,) + ds.shape[1:])
        ds[old_n:new_n] = data

    def _append_strings(self, name: str, values: List[str]) -> None:
        dtype = h5py.string_dtype(encoding="utf-8")
        arr = np.asarray(values, dtype=dtype)
        if name not in self._f:
            self._f.create_dataset(
                name,
                data=arr,
                maxshape=(None,),
                dtype=dtype,
                compression="gzip",
                compression_opts=1,
                chunks=True,
            )
            return

        ds = self._f[name]
        old_n = int(ds.shape[0])
        new_n = old_n + int(arr.shape[0])
        ds.resize((new_n,))
        ds[old_n:new_n] = arr

    def append_block(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        subject: str,
        label: str,
        slice_indices: np.ndarray,
        original_slices: int,
        crop_start: int,
        crop_end: int,
        native_width: int,
        motion_path: str,
        truth_path: str,
    ) -> None:
        n = int(x.shape[0])
        self._append_numeric("x", np.asarray(x, dtype=np.float32))
        self._append_numeric("y", np.asarray(y, dtype=np.float32))
        self._append_numeric("slice_index", np.asarray(slice_indices, dtype=np.int32))
        self._append_numeric("original_slices", np.full((n,), int(original_slices), dtype=np.int32))
        self._append_numeric("crop_start", np.full((n,), int(crop_start), dtype=np.int32))
        self._append_numeric("crop_end", np.full((n,), int(crop_end), dtype=np.int32))
        self._append_numeric("native_width", np.full((n,), int(native_width), dtype=np.int32))
        self._append_strings("subject", [subject] * n)
        self._append_strings("label", [label] * n)
        self._append_strings("motion_path", [motion_path] * n)
        self._append_strings("truth_path", [truth_path] * n)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs_from_repo(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root(), path))


def _format_path(root: str, subj: str, template: str, label: str) -> str:
    return os.path.join(root, subj, template.format(subj=subj, label=label))


def _coerce_motion_labels(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return ["" if item is None else str(item) for item in value]
    raise ValueError(f"Unsupported motion_labels type: {type(value)!r}. Expected string or list of strings.")


def _motion_labels_for_mode(data_cfg: Dict, mode: str) -> List[str]:
    for key in (f"{mode}_motion_labels", "motion_labels"):
        labels = _coerce_motion_labels(data_cfg.get(key))
        if labels:
            seen = set()
            ordered: List[str] = []
            for label in labels:
                if label in seen:
                    continue
                seen.add(label)
                ordered.append(label)
            return ordered

    for key in (f"{mode}_motion_label", "motion_label"):
        if key in data_cfg:
            return [str(data_cfg.get(key, ""))]

    return [""]


def _label_suffix(label: str) -> str:
    cleaned = str(label).strip()
    if not cleaned:
        return "nolabel"
    cleaned = cleaned.lstrip("_")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned)
    cleaned = cleaned.strip("-._")
    return cleaned or "nolabel"


def target_echo_count(config: Dict) -> Optional[int]:
    """Number of echoes expected by the network (``method.unet3d.input_channel / 2``)."""
    unet = config.get("method", {}).get("unet3d", {})
    input_channel = unet.get("input_channel")
    if input_channel:
        return int(input_channel) // 2
    return None


def align_echo_count(arr: np.ndarray, target_echo: Optional[int], subj: str, tag: str) -> np.ndarray:
    if target_echo is None:
        return arr
    if arr.shape[-1] == target_echo:
        return arr
    if arr.shape[-1] > target_echo:
        return arr[..., :target_echo]
    raise ValueError(
        f"Subject {subj} has {tag} echo count={arr.shape[-1]} smaller than required target={target_echo}."
    )


def _config_signature(
    *,
    dataset: str,
    mode: str,
    subjects: List[str],
    labels: List[str],
    slice_rng,
    norm_method: str,
    pad_to_192: bool,
    target_echo: Optional[int],
    extra: Dict,
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": dataset,
        "mode": mode,
        "subjects": list(subjects),
        "labels": list(labels),
        "slice_rng": slice_rng,
        "norm_method": norm_method,
        "pad_to_192": bool(pad_to_192),
        "target_echo": target_echo,
        "extra": extra,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def build_cache_for_mode(
    config: Dict,
    mode: Literal["train", "valid", "test"],
    cache_root: str = "cache/memoc",
    rebuild_cache: bool = False,
) -> List[str]:
    """Build the slice cache for ``mode`` (if missing) and return the shard file paths."""
    dataset = str(config.get("setting", {}).get("dataset", "mri_3decho"))
    if dataset != "mri_3decho":
        raise ValueError(f"ME-MoC supports setting.dataset='mri_3decho' only, got {dataset!r}.")

    data_cfg = config["data"]
    truth_root = _abs_from_repo(data_cfg.get("truth_root", "data/truth"))
    motion_root = _abs_from_repo(data_cfg.get("motion_root", "data/motion"))
    labels = _motion_labels_for_mode(data_cfg, mode)

    subjects = data_cfg.get(f"{mode}_subjects")
    if subjects is None:
        subjects = data_cfg.get(f"{mode}_subject")
    if not subjects:
        raise ValueError(f"No subjects configured for mode={mode} (data.{mode}_subjects).")

    slice_rng = config["setting"].get("slice_rng", None)
    norm_method = str(data_cfg.get("norm", "none"))
    pad_to_192 = bool(config["setting"].get("pad_width_to_192", True))

    truth_template = str(data_cfg.get("truth_mat_name", "ima_comb_{subj}.mat"))
    motion_template = str(data_cfg.get("motion_mat_name", "ima_comb_{subj}{label}.mat"))
    truth_keys = tuple(data_cfg.get("truth_var_candidates", ["ima_comb1", "ima_comb"]))
    motion_keys = tuple(data_cfg.get("motion_var_candidates", ["ima_comb1", "ima_comb"]))

    extra_signature = {
        "truth_root": truth_root,
        "motion_root": motion_root,
        "truth_template": truth_template,
        "motion_template": motion_template,
        "truth_keys": truth_keys,
        "motion_keys": motion_keys,
    }

    target_echo = target_echo_count(config)
    cache_paths = CachePaths(cache_root=_abs_from_repo(cache_root))
    signature = _config_signature(
        dataset=dataset,
        mode=mode,
        subjects=list(subjects),
        labels=list(labels),
        slice_rng=slice_rng,
        norm_method=norm_method,
        pad_to_192=pad_to_192,
        target_echo=target_echo,
        extra=extra_signature,
    )

    shard_files = [
        cache_paths.shard_cache(dataset, mode, signature, _label_suffix(label))
        for label in labels
    ]
    if not rebuild_cache and all(os.path.exists(path) for path in shard_files):
        return shard_files

    for path in shard_files:
        ensure_dir(os.path.dirname(path))
        if os.path.exists(path):
            os.remove(path)

    writers = {
        label: _ShardWriter(
            path=cache_paths.shard_cache(dataset, mode, signature, _label_suffix(label)),
            dataset=dataset,
            mode=mode,
            label=label,
            signature=signature,
        )
        for label in labels
    }

    try:
        for subj in subjects:
            for label in labels:
                truth_path = _format_path(truth_root, subj, truth_template, label)
                motion_path = _format_path(motion_root, subj, motion_template, label)

                truth = load_mat_var(truth_path, preferred_keys=truth_keys)
                motion = load_mat_var(motion_path, preferred_keys=motion_keys)

                truth_nxye_full = swap_to_nxye(truth)
                motion_nxye_full = swap_to_nxye(motion)

                original_slices = int(truth_nxye_full.shape[0])
                native_width = int(motion_nxye_full.shape[2])
                crop_start, crop_end = resolve_slice_bounds(original_slices, slice_rng)

                # Each volume is normalised by its own reference value.
                truth_nxye_full = norm_data(truth_nxye_full, norm_method)
                motion_nxye_full = norm_data(motion_nxye_full, norm_method)

                truth_nxye_full = align_echo_count(truth_nxye_full, target_echo, subj, "truth")
                motion_nxye_full = align_echo_count(motion_nxye_full, target_echo, subj, "motion")

                truth_nxye = crop_slices(truth_nxye_full, slice_rng)
                motion_nxye = crop_slices(motion_nxye_full, slice_rng)

                if pad_to_192:
                    truth_nxye = pad_width_190_to_192(truth_nxye)
                    motion_nxye = pad_width_190_to_192(motion_nxye)

                x = make_channels(motion_nxye, output_type="complex", new_dim=True).astype(np.float32)
                y = make_channels(truth_nxye, output_type="complex", new_dim=True).astype(np.float32)

                slice_indices = np.arange(crop_start, crop_end, dtype=np.int32)
                writers[label].append_block(
                    x=x,
                    y=y,
                    subject=subj,
                    label=label,
                    slice_indices=slice_indices,
                    original_slices=original_slices,
                    crop_start=crop_start,
                    crop_end=crop_end,
                    native_width=native_width,
                    motion_path=motion_path,
                    truth_path=truth_path,
                )
    finally:
        for writer in writers.values():
            writer.close()

    return shard_files
