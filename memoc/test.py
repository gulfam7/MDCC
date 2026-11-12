"""Run ME-MoC inference, with optional paired evaluation against motion-free references.

Motion-only inference (no ground truth), e.g. one volume at
``data/motion/<subj>/ima_comb_<subj>.mat``:

    python -m memoc.test --config weights/memoc/config.json \
        --weights weights/memoc/best-psnr.h5 --subjects <subj> --motion_only \
        --motion_root data/motion --motion_label ""

Paired evaluation on ``data.test_subjects`` (builds a test cache from
``data.truth_root`` and ``data.motion_root``):

    python -m memoc.test --config weights/memoc/config.json \
        --weights weights/memoc/best-psnr.h5 --motion_label _nrand2-5_snr3d_mid_r2

The network corrects the slices in ``setting.slice_rng`` (use
``--no_slice_crop`` for all slices). Corrected slices are written back into
the full volume, the width is restored to the native size, and intensities
are restored to the input scale before saving ``<subj>__<label>_corrected.mat``.
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tensorflow as tf

from .cache_builder import align_echo_count, build_cache_for_mode, target_echo_count
from .mat_io import load_mat_var
from .models import build_memoc_from_config
from .preprocess import (
    crop_slices,
    denorm_data,
    make_channels,
    norm_data,
    norm_data_params,
    pad_width_190_to_192,
    resolve_slice_bounds,
    swap_to_hwse,
    swap_to_nxye,
)

VARIANT = "attention_phys"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs_from_repo(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root(), path))


def _parse_subjects(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [s.strip() for s in value.split(",")]
    items = [s for s in items if s]
    return items if items else None


def _resolve_test_subjects(config: Dict, subject_override: Optional[List[str]]) -> List[str]:
    if subject_override:
        return list(subject_override)
    data_cfg = config.get("data", {})
    for key in ("test_subjects", "test_subject"):
        val = data_cfg.get(key)
        if val:
            return list(val)
    raise ValueError("Could not determine test subjects. Set data.test_subjects or use --subjects subj1,subj2,...")


def _with_test_subjects(config: Dict, test_subjects: List[str]) -> Dict:
    out = deepcopy(config)
    out.setdefault("data", {})
    out["data"]["test_subjects"] = list(test_subjects)
    return out


def _with_slice_override(
    config: Dict,
    no_slice_crop: bool,
    slice_rng: Optional[Sequence[int]] = None,
) -> Dict:
    out = deepcopy(config)
    if no_slice_crop:
        out.setdefault("setting", {})
        out["setting"]["slice_rng"] = None
    elif slice_rng is not None:
        if len(slice_rng) != 2 or int(slice_rng[0]) < 0 or int(slice_rng[1]) <= int(slice_rng[0]):
            raise ValueError("--slice_rng requires zero-based half-open bounds START END with 0 <= START < END")
        out.setdefault("setting", {})
        out["setting"]["slice_rng"] = [int(slice_rng[0]), int(slice_rng[1])]
    return out


def _with_data_root_overrides(
    config: Dict,
    motion_root: Optional[str],
    truth_root: Optional[str],
    motion_mat_name: Optional[str] = None,
) -> Dict:
    out = deepcopy(config)
    out.setdefault("data", {})
    if motion_root:
        out["data"]["motion_root"] = motion_root
    if truth_root:
        out["data"]["truth_root"] = truth_root
    if motion_mat_name:
        if "{subj}" not in motion_mat_name:
            raise ValueError("--motion_mat_name must contain the {subj} placeholder")
        out["data"]["motion_mat_name"] = motion_mat_name
    return out


def _with_motion_label_override(config: Dict, motion_label: Optional[str]) -> Dict:
    if motion_label is None:
        return config
    out = deepcopy(config)
    out.setdefault("data", {})
    out["data"]["motion_label"] = motion_label
    out["data"]["test_motion_label"] = motion_label
    out["data"]["test_motion_labels"] = [motion_label]
    return out


def _magnitude(t: np.ndarray) -> np.ndarray:
    return np.abs(t[..., 0] + 1j * t[..., 1])


def _phase_cosine_error_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask_threshold: float = 0.03,
    start_echo: int = 0,
    end_echo: Optional[int] = None,
) -> float:
    tr = y_true[..., 0:1].astype(np.float32)
    ti = y_true[..., 1:2].astype(np.float32)
    pr = y_pred[..., 0:1].astype(np.float32)
    pi = y_pred[..., 1:2].astype(np.float32)
    tmag = np.sqrt(tr * tr + ti * ti + 1e-8)
    pmag = np.sqrt(pr * pr + pi * pi + 1e-8)
    cos = (tr * pr + ti * pi) / (tmag * pmag + 1e-8)
    echo1 = tmag[:, :, :, 0:1, :]
    scale = np.max(echo1, axis=(1, 2), keepdims=True)
    mask = (echo1 >= np.maximum(scale * float(mask_threshold), 1e-8)).astype(np.float32)
    cos = cos[:, :, :, int(start_echo) : end_echo, :]
    mask = np.repeat(mask, repeats=cos.shape[3], axis=3)
    return float(np.sum((1.0 - cos) * mask) / (np.sum(mask) + 1e-8))


def _snr_np(y_true_mag: np.ndarray, y_pred_mag: np.ndarray) -> float:
    y_true_flat = tf.reshape(tf.convert_to_tensor(y_true_mag, dtype=tf.float32), [-1])
    y_pred_flat = tf.reshape(tf.convert_to_tensor(y_pred_mag, dtype=tf.float32), [-1])
    num = tf.norm(y_true_flat)
    den = tf.norm(y_true_flat - y_pred_flat)
    snr = 20.0 * (tf.math.log(num / (den + 1e-12)) / tf.math.log(tf.constant(10.0, dtype=y_true_flat.dtype)))
    return float(snr.numpy())


def _mag_psnr_ssim(y_true_mag: np.ndarray, y_pred_mag: np.ndarray, max_val: float) -> Tuple[float, float]:
    y_true_mag = y_true_mag.astype(np.float32)
    y_pred_mag = y_pred_mag.astype(np.float32)
    psnr_vals = tf.image.psnr(y_pred_mag, y_true_mag, max_val=float(max_val))
    ssim_vals = tf.image.ssim(y_pred_mag, y_true_mag, max_val=float(max_val))
    return float(tf.reduce_mean(psnr_vals).numpy()), float(tf.reduce_mean(ssim_vals).numpy())


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, max_val: float, phase_mask_threshold: float) -> Dict[str, float]:
    """Quick slice-level metrics on normalised magnitudes (overall, echo 1, echoes 2-E, per echo)."""
    y_true_mag = _magnitude(y_true).astype(np.float32)
    y_pred_mag = _magnitude(y_pred).astype(np.float32)

    psnr, ssim = _mag_psnr_ssim(y_true_mag, y_pred_mag, max_val=max_val)
    snr = _snr_np(y_true_mag, y_pred_mag)
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    metrics: Dict[str, float] = {
        "psnr": psnr,
        "ssim": ssim,
        "snr": snr,
        "mse": float(mse.numpy()),
        "phase_cosine_error": _phase_cosine_error_np(y_true, y_pred, mask_threshold=phase_mask_threshold),
    }

    echo_count = int(y_true_mag.shape[-1])
    if echo_count >= 1:
        e1_true = y_true_mag[..., 0:1]
        e1_pred = y_pred_mag[..., 0:1]
        e1_psnr, e1_ssim = _mag_psnr_ssim(e1_true, e1_pred, max_val=max_val)
        metrics.update({
            "echo1_psnr": e1_psnr,
            "echo1_ssim": e1_ssim,
            "echo1_snr": _snr_np(e1_true, e1_pred),
        })
    if echo_count > 1:
        rest_true = y_true_mag[..., 1:]
        rest_pred = y_pred_mag[..., 1:]
        rest_psnr, rest_ssim = _mag_psnr_ssim(rest_true, rest_pred, max_val=max_val)
        metrics.update({
            "echo2_10_psnr": rest_psnr,
            "echo2_10_ssim": rest_ssim,
            "echo2_10_snr": _snr_np(rest_true, rest_pred),
            "echo2_10_phase_cosine_error": _phase_cosine_error_np(
                y_true,
                y_pred,
                mask_threshold=phase_mask_threshold,
                start_echo=1,
                end_echo=None,
            ),
        })

    for echo_idx in range(echo_count):
        echo_true = y_true_mag[..., echo_idx : echo_idx + 1]
        echo_pred = y_pred_mag[..., echo_idx : echo_idx + 1]
        e_psnr, e_ssim = _mag_psnr_ssim(echo_true, echo_pred, max_val=max_val)
        prefix = f"echo_{echo_idx + 1:02d}"
        metrics[f"{prefix}_psnr"] = e_psnr
        metrics[f"{prefix}_ssim"] = e_ssim
        metrics[f"{prefix}_snr"] = _snr_np(echo_true, echo_pred)

    return metrics


def _save_corrected_mat(out_mat_path: str, corrected_nxye_complex: np.ndarray, key: str = "ima_comb") -> None:
    corrected_hwse = swap_to_hwse(corrected_nxye_complex)
    try:
        from scipy.io import savemat  # type: ignore

        savemat(out_mat_path, {key: corrected_hwse})
    except Exception as e:
        print(f"[ME-MoC] warning: failed to save MAT {out_mat_path}: {e}")


def _decode_string_array(values) -> List[str]:
    out: List[str] = []
    for item in values:
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _resolve_motion_path(config: Dict, subj: str, motion_label: Optional[str]) -> str:
    data_cfg = config.get("data", {})
    motion_root = _abs_from_repo(str(data_cfg.get("motion_root", "data/motion")))
    label = str(motion_label if motion_label is not None else data_cfg.get("motion_label", ""))
    motion_template = str(data_cfg.get("motion_mat_name", "ima_comb_{subj}{label}.mat"))
    candidate_dirs = [os.path.join(motion_root, subj), motion_root]

    if label:
        exact_name = motion_template.format(subj=subj, label=label)
        exact_matches = sorted({os.path.join(d, exact_name) for d in candidate_dirs if os.path.exists(os.path.join(d, exact_name))})
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise ValueError(f"Ambiguous labeled motion file for subject={subj}, label={label}: {exact_matches}")
        raise FileNotFoundError(
            f"Could not find exact motion file for subject={subj}, label={label}. "
            f"Expected {exact_name} under {candidate_dirs}."
        )

    template_no_label = motion_template.format(subj=subj, label="")
    fallback_exact = f"ima_comb_{subj}.mat"
    exact_candidates = []
    for d in candidate_dirs:
        for name in (template_no_label, fallback_exact):
            path = os.path.join(d, name)
            if os.path.exists(path):
                exact_candidates.append(path)
    exact_candidates = sorted(set(exact_candidates))
    if len(exact_candidates) == 1:
        return exact_candidates[0]
    if len(exact_candidates) > 1:
        raise ValueError(f"Ambiguous unlabeled exact motion files for subject={subj}: {exact_candidates}")

    glob_matches: List[str] = []
    for d in candidate_dirs:
        glob_matches.extend(glob(os.path.join(d, f"ima_comb_{subj}*.mat")))
    glob_matches = sorted(set(glob_matches))
    if len(glob_matches) == 1:
        return glob_matches[0]
    if len(glob_matches) > 1:
        raise ValueError(
            f"Ambiguous fallback motion files for subject={subj}. "
            "Pass --motion_label to select one explicitly. "
            f"Matches={glob_matches}"
        )
    raise FileNotFoundError(f"Could not find motion file for subject={subj} under {candidate_dirs}")


def _prepare_motion_volume_for_restore(
    config: Dict,
    subj: str,
    motion_path: str,
    motion_keys: Sequence[str],
    return_norm_params: bool = False,
):
    """Load, normalise, echo-align and width-pad a full motion volume (S, H, W, E)."""
    data_cfg = config.get("data", {})
    norm_method = str(data_cfg.get("norm", "none"))
    target_echo = target_echo_count(config)
    motion = load_mat_var(motion_path, preferred_keys=tuple(motion_keys))
    motion_nxye = swap_to_nxye(motion)
    native_width = int(motion_nxye.shape[2])
    norm_offset, norm_scale = norm_data_params(motion_nxye, norm_method)
    motion_nxye = norm_data(motion_nxye, norm_method)
    motion_nxye = align_echo_count(motion_nxye, target_echo, subj, "motion")
    if bool(config.get("setting", {}).get("pad_width_to_192", True)):
        motion_nxye = pad_width_190_to_192(motion_nxye)
    if return_norm_params:
        return motion_nxye, native_width, norm_offset, norm_scale
    return motion_nxye, native_width


def _load_motion_only_subject(config: Dict, subj: str, motion_label: Optional[str], *, return_slice_bounds: bool = False):
    data_cfg = config.get("data", {})
    motion_keys = tuple(data_cfg.get("motion_var_candidates", ["ima_comb", "ima_comb1"]))
    motion_path = _resolve_motion_path(config, subj, motion_label=motion_label)
    full_motion_nxye, _native_width = _prepare_motion_volume_for_restore(config, subj, motion_path, motion_keys)

    slice_rng = config.get("setting", {}).get("slice_rng", None)
    bounds = resolve_slice_bounds(int(full_motion_nxye.shape[0]), slice_rng)
    if bounds[0] >= bounds[1]:
        raise ValueError(f"Empty correction slice range {bounds} for subject {subj}")
    cropped = crop_slices(full_motion_nxye, slice_rng) if slice_rng is not None else full_motion_nxye

    x = make_channels(cropped, output_type="complex", new_dim=True).astype(np.float32)
    print(
        f"[ME-MoC] subject={subj} slices before/after crop: "
        f"{full_motion_nxye.shape[0]} -> {cropped.shape[0]} (slice_rng={slice_rng})"
    )
    if return_slice_bounds:
        return x, motion_path, bounds
    return x, motion_path


def _group_cached_subjects(cache_files: Sequence[str]) -> List[Dict]:
    groups: List[Dict] = []
    for cache_file in cache_files:
        with h5py.File(cache_file, "r") as f:
            subjects = _decode_string_array(f["subject"][:])
            labels = _decode_string_array(f["label"][:])
            motion_paths = _decode_string_array(f["motion_path"][:])
            truth_paths = _decode_string_array(f["truth_path"][:])
            slice_indices = f["slice_index"][:].astype(np.int32)
            crop_start = f["crop_start"][:].astype(np.int32)
            crop_end = f["crop_end"][:].astype(np.int32)
            original_slices = f["original_slices"][:].astype(np.int32)
            native_width = f["native_width"][:].astype(np.int32)

            index_map: Dict[Tuple[str, str], List[int]] = {}
            for i, key in enumerate(zip(subjects, labels)):
                index_map.setdefault((key[0], key[1]), []).append(i)

            for (subject, label), idxs in index_map.items():
                idx_arr = np.asarray(idxs, dtype=np.int32)
                order = np.argsort(slice_indices[idx_arr])
                idx_arr = idx_arr[order]
                groups.append(
                    {
                        "subject": subject,
                        "label": label,
                        "motion_path": motion_paths[idx_arr[0]],
                        "truth_path": truth_paths[idx_arr[0]],
                        "slice_indices": slice_indices[idx_arr],
                        "crop_start": int(crop_start[idx_arr[0]]),
                        "crop_end": int(crop_end[idx_arr[0]]),
                        "original_slices": int(original_slices[idx_arr[0]]),
                        "native_width": int(native_width[idx_arr[0]]),
                        "x": f["x"][idx_arr].astype(np.float32),
                        "y": f["y"][idx_arr].astype(np.float32),
                        "cache_file": cache_file,
                    }
                )
    groups.sort(key=lambda row: (row["subject"], row["label"]))
    return groups


def _restore_full_prediction(
    *,
    config: Dict,
    subject: str,
    motion_path: str,
    pred_cropped: np.ndarray,
    crop_start: int,
    crop_end: int,
    native_width_hint: Optional[int],
    motion_keys: Sequence[str],
    restore_native_width: bool,
):
    """Insert corrected slices into the (normalised) full input volume.

    Slices outside the correction crop are kept from the input. Returns the
    restored (S, H, W, E, 2) array, the native width and the normalisation
    parameters needed to map back to the input intensity scale.
    """
    full_motion_nxye, inferred_native_width, norm_offset, norm_scale = _prepare_motion_volume_for_restore(
        config,
        subject,
        motion_path,
        motion_keys,
        return_norm_params=True,
    )
    restored = make_channels(full_motion_nxye, output_type="complex", new_dim=True).astype(np.float32)
    restored[int(crop_start):int(crop_end)] = pred_cropped.astype(np.float32)
    native_width = int(native_width_hint if native_width_hint is not None else inferred_native_width)

    auto_restore_legacy_width = native_width == 190 and restored.shape[2] != native_width
    force_restore_native_width = bool(restore_native_width) and restored.shape[2] != native_width
    if auto_restore_legacy_width or force_restore_native_width:
        target_w = min(int(native_width), int(restored.shape[2]))
        restored = restored[:, :, :target_w, ...]
    return restored, native_width, norm_offset, norm_scale


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in values if x is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _safe_filename_part(value: str, fallback: str = "unknown") -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    safe = safe.strip("._-")
    return safe or fallback


def _aggregate_per_label(rows: List[Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    labels = sorted({str(row.get("label", "")) for row in rows})
    out: Dict[str, Dict[str, Optional[float]]] = {}
    metric_keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.floating))
        and key not in {"crop_start", "crop_end", "native_width"}
    })
    for label in labels:
        subset = [row for row in rows if str(row.get("label", "")) == label]
        label_summary: Dict[str, Optional[float]] = {"num_subjects": len(subset)}
        for key in metric_keys:
            label_summary[f"mean_{key}"] = _mean_or_none([row.get(key) for row in subset])
        out[label] = label_summary
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="ME-MoC inference and paired test runner.")
    parser.add_argument("--config", required=True, help="Config JSON used for training, e.g. weights/memoc/config.json")
    parser.add_argument("--weights", required=True, help="Model weights (.h5), e.g. weights/memoc/best-psnr.h5")
    parser.add_argument("--subjects", default=None, help="Comma-separated subject IDs (default: data.test_subjects)")
    parser.add_argument("--cache_root", default="cache/memoc")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--motion_only", action="store_true", help="Run inference without ground truth (no paired metrics).")
    parser.add_argument("--no_slice_crop", action="store_true", help="Ignore setting.slice_rng and correct all slices.")
    parser.add_argument(
        "--slice_rng", nargs=2, type=int, default=None, metavar=("START", "END"),
        help="Override setting.slice_rng with zero-based half-open bounds. Cannot be combined with --no_slice_crop.",
    )
    parser.add_argument("--motion_root", default=None, help="Override data.motion_root.")
    parser.add_argument("--truth_root", default=None, help="Override data.truth_root (paired testing).")
    parser.add_argument("--motion_label", default=None, help="Motion file label, e.g. _nrand2-5_snr3d_mid_r2 ('' for none).")
    parser.add_argument(
        "--motion_mat_name", default=None,
        help="Override data.motion_mat_name, e.g. ima_comb_{subj}{label}.mat.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size (slices)")
    parser.add_argument("--output_dir", default="results/memoc", help="Directory for outputs")
    parser.add_argument("--mat_key", default="ima_comb", help="MAT variable name of the corrected output")
    parser.add_argument(
        "--save_normalized_mat",
        action="store_true",
        help="Save the corrected MAT in network-normalised units instead of the input intensity scale.",
    )
    parser.add_argument(
        "--restore_native_width",
        action="store_true",
        help="Crop the saved output back to the native input width for any width mismatch "
             "(width 190 is always restored).",
    )
    args = parser.parse_args()

    with open(_abs_from_repo(args.config), "r") as f:
        config = json.load(f)
    requested = str(config.get("train", {}).get("variant", VARIANT)).lower()
    if requested != VARIANT:
        raise ValueError(f"This repository provides the '{VARIANT}' ME-MoC model only; config requests {requested!r}.")
    if config.get("setting", {}).get("dataset", "mri_3decho") != "mri_3decho":
        raise ValueError("ME-MoC supports setting.dataset='mri_3decho' only.")

    subjects = _resolve_test_subjects(config, _parse_subjects(args.subjects))
    config_test = _with_test_subjects(config, subjects)
    if args.no_slice_crop and args.slice_rng is not None:
        raise ValueError("--no_slice_crop and --slice_rng are mutually exclusive")
    config_test = _with_slice_override(
        config_test, no_slice_crop=bool(args.no_slice_crop), slice_rng=args.slice_rng
    )
    config_test = _with_data_root_overrides(
        config_test, motion_root=args.motion_root, truth_root=args.truth_root,
        motion_mat_name=args.motion_mat_name,
    )
    config_test = _with_motion_label_override(config_test, args.motion_label)

    cache_groups: List[Dict] = []
    if not args.motion_only:
        cache_root = _abs_from_repo(args.cache_root)
        cache_files = build_cache_for_mode(
            config_test,
            mode="test",
            cache_root=cache_root,
            rebuild_cache=bool(args.rebuild_cache),
        )
        if not cache_files:
            raise ValueError("No test cache files were created/found.")
        cache_groups = _group_cached_subjects(cache_files)
        if not cache_groups:
            raise ValueError("Test cache was created but no grouped subject records were found.")
        sample_shape = tuple(cache_groups[0]["x"].shape[1:])
    else:
        x0, _ = _load_motion_only_subject(config_test, subjects[0], motion_label=args.motion_label)
        sample_shape = tuple(x0.shape[1:])

    print(f"[ME-MoC] building model sample_shape={sample_shape}", flush=True)
    model = build_memoc_from_config(config_test, input_shape=sample_shape)
    print(f"[ME-MoC] model built: parameters={model.count_params():,}", flush=True)

    weights_path = _abs_from_repo(args.weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    print(f"[ME-MoC] loading weights: {weights_path}", flush=True)
    model.load_weights(weights_path)

    output_root = _abs_from_repo(args.output_dir)
    run_dir = os.path.join(output_root, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_ME-MoC")
    os.makedirs(run_dir, exist_ok=True)

    psnr_max_val = float(config.get("train", {}).get("psnr_max_val", 1.0))
    physics_cfg = config_test.get("train", {}).get("physics", {})
    phase_mask_threshold = float(physics_cfg.get("mask_threshold", 0.03))
    motion_keys = tuple(config_test.get("data", {}).get("motion_var_candidates", ["ima_comb", "ima_comb1"]))
    per_subject: List[Dict] = []

    iterable = cache_groups if cache_groups else subjects
    for item in iterable:
        metrics = None
        if cache_groups:
            group = item
            subject = str(group["subject"])
            label = str(group["label"])
            motion_path = str(group["motion_path"])
            x = group["x"]
            y = group["y"]
            crop_start = int(group["crop_start"])
            crop_end = int(group["crop_end"])
            native_width_hint = int(group["native_width"])
            cache_file = str(group["cache_file"])
        else:
            subject = str(item)
            label = str(
                args.motion_label
                if args.motion_label is not None
                else config_test.get("data", {}).get("motion_label", "")
            )
            x, motion_path, (crop_start, crop_end) = _load_motion_only_subject(
                config_test, subject, motion_label=args.motion_label, return_slice_bounds=True
            )
            y = None
            cache_file = None
            native_width_hint = None

        print(
            f"[ME-MoC] subject={subject} starting prediction: slices={x.shape[0]} "
            f"batch_size={int(args.batch_size)}",
            flush=True,
        )
        pred = model.predict(x, batch_size=int(args.batch_size), verbose=0).astype(np.float32)
        restored_pred, native_width, norm_offset, norm_scale = _restore_full_prediction(
            config=config_test,
            subject=subject,
            motion_path=motion_path,
            pred_cropped=pred,
            crop_start=crop_start,
            crop_end=crop_end,
            native_width_hint=native_width_hint,
            motion_keys=motion_keys,
            restore_native_width=bool(args.restore_native_width),
        )

        if y is not None:
            metrics = _compute_metrics(
                y_true=y,
                y_pred=pred,
                max_val=psnr_max_val,
                phase_mask_threshold=phase_mask_threshold,
            )

        safe_label = _safe_filename_part(label or "nolabel", fallback="nolabel")
        base_name = f"{subject}__{safe_label}"
        out_file = os.path.join(run_dir, f"{base_name}.h5")
        out_mat_file = os.path.join(run_dir, f"{base_name}_corrected.mat")
        with h5py.File(out_file, "w") as f:
            f.attrs["subject"] = subject
            f.attrs["label"] = label
            f.attrs["model"] = "ME-MoC"
            f.attrs["weights"] = weights_path
            f.attrs["motion_path"] = motion_path
            f.attrs["crop_start"] = int(crop_start)
            f.attrs["crop_end"] = int(crop_end)
            f.attrs["original_slices"] = int(restored_pred.shape[0])
            f.attrs["native_width"] = int(native_width)
            f.attrs["norm_method"] = str(config_test.get("data", {}).get("norm", "none"))
            f.attrs["mat_output_scale"] = "normalized" if args.save_normalized_mat else "input_native"
            f.attrs["norm_scale_shape"] = str(np.shape(norm_scale))
            if np.size(norm_scale) == 1:
                f.attrs["norm_scale"] = float(np.asarray(norm_scale).reshape(-1)[0])
            if metrics is not None:
                for key, value in metrics.items():
                    f.attrs[key] = float(value)
            f.create_dataset("x", data=x, compression="gzip", compression_opts=1, chunks=True)
            if y is not None:
                f.create_dataset("y", data=y, compression="gzip", compression_opts=1, chunks=True)
            f.create_dataset("pred_cropped", data=pred, compression="gzip", compression_opts=1, chunks=True)

        pred_complex = (restored_pred[..., 0] + 1j * restored_pred[..., 1]).astype(np.complex64)
        if not args.save_normalized_mat:
            pred_complex = denorm_data(pred_complex, norm_offset, norm_scale).astype(np.complex64)
        _save_corrected_mat(out_mat_file, pred_complex, key=str(args.mat_key))

        row = {
            "subject": subject,
            "label": label,
            "cache_file": cache_file,
            "motion_path": motion_path,
            "output_file": out_file,
            "corrected_mat": out_mat_file,
            "crop_start": int(crop_start),
            "crop_end": int(crop_end),
            "native_width": int(native_width),
        }
        if metrics is not None:
            row.update(metrics)
            print(
                f"[ME-MoC] subject={subject} label={label} psnr={metrics['psnr']:.4f} "
                f"ssim={metrics['ssim']:.4f} snr={metrics['snr']:.4f} "
                f"echo1_psnr={metrics.get('echo1_psnr', float('nan')):.4f} "
                f"echo2_10_psnr={metrics.get('echo2_10_psnr', float('nan')):.4f} "
                f"phase_cosine_error={metrics['phase_cosine_error']:.6f}"
            )
        else:
            print(f"[ME-MoC] subject={subject} label={label} inference complete. corrected={out_mat_file}", flush=True)
        per_subject.append(row)

    summary = {
        "model": "ME-MoC",
        "variant": VARIANT,
        "weights": weights_path,
        "motion_root": config_test.get("data", {}).get("motion_root"),
        "truth_root": config_test.get("data", {}).get("truth_root"),
        "subjects": subjects,
        "num_subjects": len(per_subject),
        "parameters": {
            "total": int(model.count_params()),
            "trainable": int(sum(np.prod(v.shape) for v in model.trainable_weights)),
            "non_trainable": int(sum(np.prod(v.shape) for v in model.non_trainable_weights)),
        },
        "psnr_max_val": psnr_max_val,
        "motion_only": bool(args.motion_only),
        "mean_psnr": _mean_or_none([r.get("psnr") for r in per_subject]),
        "mean_ssim": _mean_or_none([r.get("ssim") for r in per_subject]),
        "mean_snr": _mean_or_none([r.get("snr") for r in per_subject]),
        "mean_mse": _mean_or_none([r.get("mse") for r in per_subject]),
        "mean_phase_cosine_error": _mean_or_none([r.get("phase_cosine_error") for r in per_subject]),
        "mean_echo1_psnr": _mean_or_none([r.get("echo1_psnr") for r in per_subject]),
        "mean_echo1_ssim": _mean_or_none([r.get("echo1_ssim") for r in per_subject]),
        "mean_echo1_snr": _mean_or_none([r.get("echo1_snr") for r in per_subject]),
        "mean_echo2_10_psnr": _mean_or_none([r.get("echo2_10_psnr") for r in per_subject]),
        "mean_echo2_10_ssim": _mean_or_none([r.get("echo2_10_ssim") for r in per_subject]),
        "mean_echo2_10_snr": _mean_or_none([r.get("echo2_10_snr") for r in per_subject]),
        "mean_echo2_10_phase_cosine_error": _mean_or_none([r.get("echo2_10_phase_cosine_error") for r in per_subject]),
        "per_label": _aggregate_per_label(per_subject),
        "per_subject": per_subject,
    }

    summary_path = os.path.join(run_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ME-MoC] done. summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
