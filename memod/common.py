"""Configuration, path, slice-label and input helpers shared by ME-MoD scripts."""
from __future__ import annotations

import json
import os
import re
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .mat_io import load_mat_var
from .preprocess import (
    crop_slices,
    ensure_dir,
    make_channels,
    norm_data,
    pad_width_190_to_192,
    swap_to_nxye,
)

__all__ = [
    "abs_from_repo",
    "align_echo_count",
    "apply_config_defaults",
    "ensure_dir",
    "format_path",
    "input_from_nxye",
    "load_config",
    "load_mat_var",
    "load_preprocessed_input",
    "load_slice_labels_from_mat",
    "prepare_nxye",
    "real_subjects_for_mode",
    "repo_root",
    "resolve_motion_roots",
    "resolve_motion_variants",
    "sanitize_name",
    "save_json",
    "subjects_for_mode",
]


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def abs_from_repo(path: str) -> str:
    """Resolve a relative path against the repository root."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root(), path))


def save_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def apply_config_defaults(config: Dict) -> Dict:
    """Fill missing config entries with the ME-MoD settings used in the paper."""
    setting = config.setdefault("setting", {})
    train = config.setdefault("train", {})
    classification = config.setdefault("classification", {})
    model = config.setdefault("model", {})
    aggregation = classification.setdefault("aggregation", {})
    localization = classification.setdefault("localization", {})
    training = classification.setdefault("training", {})
    data_cfg = config.setdefault("data", {})

    setting.setdefault("task", "motion_classification_subject_gate")
    setting.setdefault("dataset", "mri_3decho")
    setting.setdefault("root_path", "runs/memod")
    setting.setdefault("save_folder", "ME-MoD")
    setting.setdefault("slice_rng", [29, 75])
    setting.setdefault("pad_width_to_192", True)

    train.setdefault("subject_batch_size", 4)
    train.setdefault("learning_rate", 1e-4)
    train.setdefault("train_epoch", 150)
    train.setdefault("early_stopping_patience", 50)
    train.setdefault("monitor", "val_auc")
    train.setdefault("monitor_mode", "max")
    train.setdefault("max_slices", None)

    classification.setdefault("input_representation", "complex20")
    classification.setdefault("echo_count", 10)
    classification.setdefault("class_balance", "weighted")
    classification.setdefault("decision_threshold", 0.5)
    classification.setdefault("slice_label_fallback", "dense")
    classification.setdefault(
        "real_motion_slice_label_fallback",
        classification.get("slice_label_fallback", "dense"),
    )

    aggregation.setdefault("type", "topk_mean")
    aggregation.setdefault("topk", 14)
    aggregation.setdefault("window_size", 3)
    aggregation.setdefault("logit_floor", 0.0)
    aggregation.setdefault("min_positive", 3)
    aggregation.setdefault("fallback_logit", -6.0)

    localization.setdefault("enable", True)
    localization.setdefault("threshold", 0.5)
    localization.setdefault("max_count", 12)
    localization.setdefault("max_fraction", 0.12)
    localization.setdefault("min_consecutive", 3)

    training.setdefault("subject_loss_weight", 1.0)
    training.setdefault("slice_loss_weight", 0.15)
    training.setdefault("slice_unknown_value", -1.0)

    model.setdefault("gate_hidden", 32)
    model.setdefault("echo_gate_type", "complex_pair_se")
    model.setdefault("echo_gate_eps", 1e-6)
    model.setdefault("stem_filters", 16)
    model.setdefault("stage_filters", [16, 32, 64, 128])
    model.setdefault("blocks_per_stage", 2)
    model.setdefault("embedding_dim", 128)
    model.setdefault("dropout", 0.2)
    model.setdefault("use_layer_norm", True)

    data_cfg.setdefault("truth_root", "data/truth")
    data_cfg.setdefault("motion_root", "data/motion")
    data_cfg.setdefault("motion_roots", [data_cfg.get("motion_root", "data/motion")])
    data_cfg.setdefault("motion_source_names", [])
    data_cfg.setdefault("motion_labels", [])
    data_cfg.setdefault("truth_mat_name", "ima_comb_{subj}.mat")
    data_cfg.setdefault("motion_mat_name", "ima_comb_{subj}{label}.mat")
    data_cfg.setdefault("truth_var_candidates", ["ima_comb", "ima_comb1"])
    data_cfg.setdefault("motion_var_candidates", ["ima_comb", "ima_comb1"])
    data_cfg.setdefault("norm", "echo1mean")
    data_cfg.setdefault("slice_label_key", "slice_label")
    data_cfg.setdefault("slice_index_key", "target_slices")
    data_cfg.setdefault("slice_label_sidecar_suffix", ".slice_labels.json")
    data_cfg.setdefault("slice_label_unknown_value", -1)
    data_cfg.setdefault("real_motion_root", "")
    data_cfg.setdefault("real_motion_source_name", "real_motion")
    data_cfg.setdefault("real_motion_train_subjects", [])
    data_cfg.setdefault("real_motion_valid_subjects", [])
    data_cfg.setdefault("real_motion_test_subjects", [])
    data_cfg.setdefault("real_motion_mat_name", data_cfg.get("truth_mat_name", "ima_comb_{subj}.mat"))
    data_cfg.setdefault("real_motion_var_candidates", list(data_cfg.get("motion_var_candidates", ["ima_comb", "ima_comb1"])))
    data_cfg.setdefault("real_clean_root", "")
    data_cfg.setdefault("real_clean_source_name", "real_clean")
    data_cfg.setdefault("real_clean_train_subjects", [])
    data_cfg.setdefault("real_clean_valid_subjects", [])
    data_cfg.setdefault("real_clean_test_subjects", [])
    data_cfg.setdefault("real_clean_mat_name", data_cfg.get("truth_mat_name", "ima_comb_{subj}.mat"))
    data_cfg.setdefault("real_clean_var_candidates", list(data_cfg.get("truth_var_candidates", ["ima_comb", "ima_comb1"])))
    data_cfg.setdefault("train_subjects", [])
    data_cfg.setdefault("valid_subjects", [])
    data_cfg.setdefault("test_subjects", [])

    return config


def load_config(path: str) -> Dict:
    with open(abs_from_repo(path), "r") as f:
        return apply_config_defaults(json.load(f))


def subjects_for_mode(data_cfg: Dict, mode: str) -> List[str]:
    values = data_cfg.get(f"{mode}_subjects")
    if values is None:
        values = data_cfg.get(f"{mode}_subject")
    if not values and mode == "test":
        values = data_cfg.get("valid_subjects", [])
    return [str(v) for v in list(values or [])]


def real_subjects_for_mode(data_cfg: Dict, prefix: str, mode: str) -> List[str]:
    return [str(v) for v in list(data_cfg.get(f"{prefix}_{mode}_subjects", []) or [])]


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def resolve_motion_roots(data_cfg: Dict) -> Tuple[List[str], List[str]]:
    motion_roots_cfg = data_cfg.get("motion_roots")
    if motion_roots_cfg is None:
        motion_roots_cfg = [data_cfg.get("motion_root", "data/motion")]
    motion_roots = [abs_from_repo(str(v)) for v in list(motion_roots_cfg) if str(v)]
    source_names_cfg = list(data_cfg.get("motion_source_names", []))
    source_names: List[str] = []
    for i, root in enumerate(motion_roots):
        if i < len(source_names_cfg) and str(source_names_cfg[i]):
            source_names.append(str(source_names_cfg[i]))
        else:
            source_names.append(os.path.basename(root.rstrip("/\\")) or f"motion_source_{i}")
    return motion_roots, source_names


def align_echo_count(nxye: np.ndarray, target_echo: int, subj: str, tag: str) -> np.ndarray:
    cur = int(nxye.shape[-1])
    tgt = int(target_echo)
    if cur == tgt:
        return nxye
    if cur > tgt:
        return nxye[..., :tgt]
    raise ValueError(f"{tag} for subject {subj} has {cur} echoes but ME-MoD expects {tgt}.")


def prepare_nxye(
    volume: np.ndarray,
    subj: str,
    tag: str,
    norm_method: str,
    target_echo: int,
    slice_rng,
    pad_to_192: bool,
) -> np.ndarray:
    """(H, W, S, E) volume -> normalised, echo-aligned, cropped, padded (S', H, 192, E)."""
    nxye = swap_to_nxye(volume)
    nxye = norm_data(nxye, norm_method)
    nxye = align_echo_count(nxye, target_echo, subj, tag)
    nxye = crop_slices(nxye, slice_rng)
    if pad_to_192:
        nxye = pad_width_190_to_192(nxye)
    return nxye


def input_from_nxye(nxye: np.ndarray, input_representation: str = "complex20") -> np.ndarray:
    """(S, H, W, E) complex -> (S, H, W, 2E) float32 with layout [Re(e1..eE), Im(e1..eE)]."""
    rep = str(input_representation).lower()
    if rep in ("complex", "complex20"):
        return np.asarray(make_channels(nxye, output_type="complex", new_dim=False), dtype=np.float32)
    raise ValueError(
        f"ME-MoD requires complex input (input_representation='complex20'), got {input_representation!r}"
    )


def load_slice_labels_from_mat(
    file_path: str,
    expected_slices: int,
    slice_label_key: str,
    slice_index_key: str,
    slice_rng,
    unknown_value: float,
    sidecar_suffix: str = ".slice_labels.json",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-slice labels and an availability mask for the cropped slices.

    Labels are read, in order of preference, from a JSON sidecar
    (``<mat-stem><sidecar_suffix>``, written by ``memod.label_slice_ranges``)
    or from ``slice_label`` / ``target_slices`` variables stored in the MAT
    file. Slices without a label get ``unknown_value`` and availability 0.
    """
    slice_label = np.full((int(expected_slices),), float(unknown_value), dtype=np.float32)
    slice_available = np.zeros((int(expected_slices),), dtype=np.float32)

    raw_label = None
    raw_indices = None
    raw_indices_source = None
    sidecar_suffix = str(sidecar_suffix or "").strip()
    if sidecar_suffix:
        sidecar_path = os.path.splitext(file_path)[0] + sidecar_suffix
        if os.path.exists(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if slice_label_key in payload:
                raw_label = np.asarray(payload[slice_label_key])
            if slice_index_key in payload:
                raw_indices = np.asarray(payload[slice_index_key])
                raw_indices_source = "sidecar"
            if raw_label is None and raw_indices is None:
                raise KeyError(
                    f"Slice-label sidecar {sidecar_path} must contain "
                    f"'{slice_label_key}' and/or '{slice_index_key}'."
                )

    try:
        from scipy.io import loadmat  # type: ignore

        if raw_label is None and raw_indices is None:
            mat = loadmat(file_path, verify_compressed_data_integrity=False)
            if slice_label_key in mat:
                raw_label = np.asarray(mat[slice_label_key])
            if slice_index_key in mat:
                raw_indices = np.asarray(mat[slice_index_key])
                raw_indices_source = "mat"
    except NotImplementedError:
        if raw_label is None and raw_indices is None:
            with h5py.File(file_path, "r") as f:
                if slice_label_key in f:
                    raw_label = np.asarray(f[slice_label_key][()])
                if slice_index_key in f:
                    raw_indices = np.asarray(f[slice_index_key][()])
                    raw_indices_source = "h5"
    except Exception:
        if raw_label is None and raw_indices is None:
            raw_label = None
            raw_indices = None

    if raw_label is not None:
        v = np.asarray(raw_label).reshape(-1)
        v = crop_slices(v, slice_rng) if slice_rng is not None else v
        if v.size == int(expected_slices):
            out = np.asarray(v, dtype=np.float32)
            known = np.isin(out, [0, 1])
            slice_label[known] = out[known]
            slice_available[known] = 1.0
            slice_label[~known] = out[~known]
            return slice_label, slice_available

    if raw_indices is not None:
        idx = np.asarray(raw_indices).reshape(-1).astype(np.int64)
        if idx.size > 0:
            # Sidecar target_slices are 0-based; MAT/H5 indices that never
            # contain 0 are treated as MATLAB 1-based indices.
            if raw_indices_source != "sidecar" and np.max(idx) >= 1 and np.min(idx) >= 1:
                all_look_1based = not np.any(idx == 0)
                if all_look_1based:
                    idx = idx - 1

            # Shift indices by the slice_rng crop offset.
            if slice_rng is not None:
                crop_start = 0
                if isinstance(slice_rng, (list, tuple)) and len(slice_rng) == 2:
                    crop_start = int(slice_rng[0])
                    crop_end = int(slice_rng[1])
                elif isinstance(slice_rng, dict):
                    # Dict-style slice_rng cannot be resolved without the
                    # original volume size; no adjustment is applied.
                    crop_start = 0
                    crop_end = None
                if crop_start > 0:
                    idx = idx - crop_start
                if isinstance(slice_rng, (list, tuple)) and len(slice_rng) == 2:
                    crop_end = int(slice_rng[1])
                    idx = idx[idx < (crop_end - crop_start)]

            idx = idx[(idx >= 0) & (idx < int(expected_slices))]
            if idx.size > 0:
                slice_label[:] = 0.0
                slice_available[:] = 1.0
                slice_label[idx] = 1.0
                return slice_label, slice_available

    return slice_label, slice_available


def resolve_motion_variants(
    subj: str,
    label: str,
    motion_roots: Sequence[str],
    motion_source_names: Sequence[str],
    motion_template: str,
) -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []
    for root, source_name in zip(motion_roots, motion_source_names):
        motion_path = format_path(root, subj, motion_template, label)
        if os.path.exists(motion_path):
            variants.append((motion_path, source_name))
            continue
        motion_dir = os.path.dirname(motion_path)
        motion_stem = os.path.splitext(os.path.basename(motion_path))[0]
        for candidate in sorted(glob(os.path.join(motion_dir, f"{motion_stem}*.mat"))):
            variants.append((candidate, source_name))
    return variants


def format_path(root: str, subj: str, template: str, label: str = "") -> str:
    return os.path.join(abs_from_repo(root), str(subj), template.format(subj=subj, label=label))


def load_preprocessed_input(
    input_mat: str,
    config: Dict,
    var_candidates: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Load one complex MAT volume and return the ME-MoD input bag (S', H, W, 2E)."""
    data_cfg = config.get("data", {})
    cls_cfg = config.get("classification", {})
    setting = config.get("setting", {})
    norm_method = str(data_cfg.get("norm", "none"))
    slice_rng = setting.get("slice_rng", None)
    pad_to_192 = bool(setting.get("pad_width_to_192", True))
    input_representation = str(cls_cfg.get("input_representation", "complex20")).lower()
    echo_count = int(cls_cfg.get("echo_count", 10))
    if not var_candidates:
        truth_keys = list(data_cfg.get("truth_var_candidates", []))
        motion_keys = list(data_cfg.get("motion_var_candidates", []))
        var_candidates = [*truth_keys, *motion_keys] if (truth_keys or motion_keys) else None
    volume = load_mat_var(abs_from_repo(input_mat), preferred_keys=tuple(var_candidates) if var_candidates else ("ima_comb", "ima_comb1"))
    nxye = prepare_nxye(
        volume=volume,
        subj="DEPLOY",
        tag="deploy",
        norm_method=norm_method,
        target_echo=echo_count,
        slice_rng=slice_rng,
        pad_to_192=pad_to_192,
    )
    return input_from_nxye(nxye, input_representation=input_representation)
