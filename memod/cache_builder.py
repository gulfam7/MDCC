"""Build per-volume HDF5 caches for ME-MoD training and evaluation.

One cache file is written per volume under
``<cache_root>/subject_gate/<mode>/<sample_id>.h5`` with datasets
``x`` (S, H, W, 2E), ``y_subject`` (volume label), ``y_slice`` (per-slice
labels) and ``y_slice_available`` (1 where a slice label is known).

Sources per mode:
  * paired motion-free volumes (``data.truth_root``)              -> label 0
  * simulated motion volumes for each ``data.motion_labels`` entry -> label 1
  * acquired motion volumes (``data.real_motion_*``)              -> label 1
  * acquired motion-free volumes (``data.real_clean_*``)          -> label 0
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import h5py
import numpy as np

from .common import (
    abs_from_repo,
    apply_config_defaults,
    ensure_dir,
    format_path,
    input_from_nxye,
    load_mat_var,
    load_slice_labels_from_mat,
    prepare_nxye,
    real_subjects_for_mode,
    resolve_motion_roots,
    resolve_motion_variants,
    sanitize_name,
    subjects_for_mode,
)


@dataclass(frozen=True)
class CachePaths:
    cache_root: str

    def sample_cache(self, mode: str, sample_id: str) -> str:
        return os.path.join(self.cache_root, "subject_gate", str(mode), f"{sanitize_name(sample_id)}.h5")


def _target_echo_count(config: Dict) -> int:
    return int(config.get("classification", {}).get("echo_count", 10))


def _write_sample_cache(
    cache_file: str,
    sample_id: str,
    subject: str,
    source_kind: str,
    source_name: str,
    source_path: str,
    x_arr: np.ndarray,
    y_subject: float,
    y_slice: np.ndarray,
    y_slice_available: np.ndarray,
    input_representation: str,
    mode: str,
) -> None:
    ensure_dir(os.path.dirname(cache_file))
    with h5py.File(cache_file, "w") as f:
        f.attrs.update(
            {
                "sample_id": str(sample_id),
                "subject": str(subject),
                "source_kind": str(source_kind),
                "source_name": str(source_name),
                "source_path": str(source_path),
                "mode": str(mode),
                "input_representation": str(input_representation),
            }
        )
        f.create_dataset("x", data=np.asarray(x_arr, dtype=np.float32), compression="gzip", compression_opts=1, chunks=True)
        f.create_dataset("y_subject", data=np.asarray([float(y_subject)], dtype=np.float32))
        f.create_dataset("y_slice", data=np.asarray(y_slice, dtype=np.float32), compression="gzip", compression_opts=1, chunks=True)
        f.create_dataset(
            "y_slice_available",
            data=np.asarray(y_slice_available, dtype=np.float32),
            compression="gzip",
            compression_opts=1,
            chunks=True,
        )


def _slice_labels_for_motion(
    config: Dict,
    motion_path: str,
    n_slices: int,
    fallback_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-slice labels for a motion volume.

    With ``fallback_mode='dense'`` a volume without stored slice labels is
    treated as motion-corrupted on every slice.
    """
    data_cfg = config.get("data", {})
    unknown_value = float(data_cfg.get("slice_label_unknown_value", -1))
    y_slice, y_avail = load_slice_labels_from_mat(
        file_path=motion_path,
        expected_slices=n_slices,
        slice_label_key=str(data_cfg.get("slice_label_key", "slice_label")),
        slice_index_key=str(data_cfg.get("slice_index_key", "target_slices")),
        slice_rng=config.get("setting", {}).get("slice_rng", None),
        unknown_value=unknown_value,
        sidecar_suffix=str(data_cfg.get("slice_label_sidecar_suffix", ".slice_labels.json")),
    )
    if y_slice.shape[0] != n_slices:
        y_slice = np.full((n_slices,), unknown_value, dtype=np.float32)
        y_avail = np.zeros((n_slices,), dtype=np.float32)
    if not np.any(y_avail > 0.5) and str(fallback_mode).lower() == "dense":
        y_slice = np.ones((n_slices,), dtype=np.float32)
        y_avail = np.ones((n_slices,), dtype=np.float32)
    return y_slice.astype(np.float32), y_avail.astype(np.float32)


def build_cache_for_mode(
    config: Dict,
    mode: Literal["train", "valid", "test"],
    cache_root: str = "cache/memod",
    rebuild_cache: bool = False,
) -> List[str]:
    """Build (if missing) and return the per-volume cache files for ``mode``."""
    config = apply_config_defaults(dict(config))
    data_cfg = config.get("data", {})
    cls_cfg = config.get("classification", {})
    setting = config.get("setting", {})

    input_representation = str(cls_cfg.get("input_representation", "complex20")).lower()
    target_echo = _target_echo_count(config)
    slice_rng = setting.get("slice_rng", None)
    pad_to_192 = bool(setting.get("pad_width_to_192", True))
    norm_method = str(data_cfg.get("norm", "none"))

    truth_root = abs_from_repo(data_cfg.get("truth_root", "data/truth"))
    motion_roots, motion_source_names = resolve_motion_roots(data_cfg)
    truth_template = str(data_cfg.get("truth_mat_name", "ima_comb_{subj}.mat"))
    motion_template = str(data_cfg.get("motion_mat_name", "ima_comb_{subj}{label}.mat"))
    truth_keys = tuple(data_cfg.get("truth_var_candidates", ["ima_comb", "ima_comb1"]))
    motion_keys = tuple(data_cfg.get("motion_var_candidates", ["ima_comb", "ima_comb1"]))

    cache_paths = CachePaths(cache_root=abs_from_repo(cache_root))
    created: List[str] = []

    def _prepare_input(volume: np.ndarray, subj: str, tag: str) -> np.ndarray:
        nxye = prepare_nxye(
            volume=volume,
            subj=subj,
            tag=tag,
            norm_method=norm_method,
            target_echo=target_echo,
            slice_rng=slice_rng,
            pad_to_192=pad_to_192,
        )
        return input_from_nxye(nxye, input_representation=input_representation)

    for subj in subjects_for_mode(data_cfg, mode):
        truth_path = format_path(truth_root, subj, truth_template, "")
        truth_sample_id = f"truth__{subj}"
        truth_cache = cache_paths.sample_cache(mode=mode, sample_id=truth_sample_id)
        if not (os.path.exists(truth_cache) and not rebuild_cache):
            truth = load_mat_var(truth_path, preferred_keys=truth_keys)
            x_truth = _prepare_input(truth, subj=str(subj), tag="truth")
            y_slice = np.zeros((x_truth.shape[0],), dtype=np.float32)
            y_avail = np.ones((x_truth.shape[0],), dtype=np.float32)
            _write_sample_cache(
                cache_file=truth_cache,
                sample_id=truth_sample_id,
                subject=str(subj),
                source_kind="paired_truth",
                source_name="truth",
                source_path=truth_path,
                x_arr=x_truth,
                y_subject=0.0,
                y_slice=y_slice,
                y_slice_available=y_avail,
                input_representation=input_representation,
                mode=mode,
            )
        created.append(truth_cache)

        for label in [str(v) for v in list(data_cfg.get("motion_labels", []))]:
            variants = resolve_motion_variants(
                subj=str(subj),
                label=label,
                motion_roots=motion_roots,
                motion_source_names=motion_source_names,
                motion_template=motion_template,
            )
            for motion_path, source_name in variants:
                motion_id = os.path.splitext(os.path.basename(motion_path))[0]
                motion_sample_id = f"motion__{subj}__{motion_id}"
                motion_cache = cache_paths.sample_cache(mode=mode, sample_id=motion_sample_id)
                if os.path.exists(motion_cache) and not rebuild_cache:
                    created.append(motion_cache)
                    continue
                motion = load_mat_var(motion_path, preferred_keys=motion_keys)
                x_motion = _prepare_input(motion, subj=str(subj), tag=f"motion@{source_name}")
                y_slice, y_avail = _slice_labels_for_motion(
                    config,
                    motion_path,
                    int(x_motion.shape[0]),
                    fallback_mode=str(cls_cfg.get("slice_label_fallback", "dense")),
                )
                _write_sample_cache(
                    cache_file=motion_cache,
                    sample_id=motion_sample_id,
                    subject=str(subj),
                    source_kind="paired_motion",
                    source_name=str(source_name),
                    source_path=motion_path,
                    x_arr=x_motion,
                    y_subject=1.0,
                    y_slice=y_slice,
                    y_slice_available=y_avail,
                    input_representation=input_representation,
                    mode=mode,
                )
                created.append(motion_cache)

    def _build_unpaired(
        subjects: Sequence[str],
        root: str,
        template: str,
        preferred_keys: Sequence[str],
        source_kind: str,
        source_name: str,
        y_subject_value: float,
        dense_zero_slice: bool,
    ) -> None:
        if not subjects:
            return
        if not root:
            raise ValueError(f"{source_kind} subjects provided for mode={mode}, but root is empty.")
        for subj in subjects:
            source_path = format_path(root, subj, template, "")
            sample_id = f"{source_kind}__{subj}"
            cache_file = cache_paths.sample_cache(mode=mode, sample_id=sample_id)
            if os.path.exists(cache_file) and not rebuild_cache:
                created.append(cache_file)
                continue
            volume = load_mat_var(source_path, preferred_keys=tuple(preferred_keys))
            x_arr = _prepare_input(volume, subj=str(subj), tag=source_kind)
            if dense_zero_slice:
                y_slice = np.zeros((x_arr.shape[0],), dtype=np.float32)
                y_avail = np.ones((x_arr.shape[0],), dtype=np.float32)
            else:
                y_slice, y_avail = _slice_labels_for_motion(
                    config,
                    source_path,
                    int(x_arr.shape[0]),
                    fallback_mode=str(
                        cls_cfg.get(
                            "real_motion_slice_label_fallback",
                            cls_cfg.get("slice_label_fallback", "dense"),
                        )
                    ),
                )
                if not np.any(y_avail > 0.5):
                    unknown = float(data_cfg.get("slice_label_unknown_value", -1))
                    y_slice = np.full((x_arr.shape[0],), unknown, dtype=np.float32)
                    y_avail = np.zeros((x_arr.shape[0],), dtype=np.float32)
            _write_sample_cache(
                cache_file=cache_file,
                sample_id=sample_id,
                subject=str(subj),
                source_kind=source_kind,
                source_name=source_name,
                source_path=source_path,
                x_arr=x_arr,
                y_subject=y_subject_value,
                y_slice=y_slice,
                y_slice_available=y_avail,
                input_representation=input_representation,
                mode=mode,
            )
            created.append(cache_file)

    _build_unpaired(
        subjects=real_subjects_for_mode(data_cfg, "real_motion", mode),
        root=str(data_cfg.get("real_motion_root", "")),
        template=str(data_cfg.get("real_motion_mat_name", truth_template)),
        preferred_keys=list(data_cfg.get("real_motion_var_candidates", motion_keys)),
        source_kind="real_motion_unpaired",
        source_name=str(data_cfg.get("real_motion_source_name", "real_motion")),
        y_subject_value=1.0,
        dense_zero_slice=False,
    )
    _build_unpaired(
        subjects=real_subjects_for_mode(data_cfg, "real_clean", mode),
        root=str(data_cfg.get("real_clean_root", "")),
        template=str(data_cfg.get("real_clean_mat_name", truth_template)),
        preferred_keys=list(data_cfg.get("real_clean_var_candidates", truth_keys)),
        source_kind="real_clean_unpaired",
        source_name=str(data_cfg.get("real_clean_source_name", "real_clean")),
        y_subject_value=0.0,
        dense_zero_slice=True,
    )
    return created
