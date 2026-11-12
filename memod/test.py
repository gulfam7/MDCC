"""Evaluate or deploy ME-MoD.

Sub-commands (run from the repository root):

    # one volume -> motion / motion_free decision and per-slice scores
    python -m memod.test deploy --config weights/memod/config.json \
        --weights weights/memod/best_weights.h5 --input_mat path/to/ima_comb_<subj>.mat

    # every ima_comb_*.mat in a folder (and its immediate subfolders)
    python -m memod.test deploy_folder --config weights/memod/config.json \
        --weights weights/memod/best_weights.h5 --input_root path/to/folder --output_json report.json

    # labelled split from the config (uses the cache builder)
    python -m memod.test eval --config configs/memod.json \
        --weights weights/memod/best_weights.h5 --split valid

The decision threshold is taken from ``--threshold`` if given, otherwise from
``decision_threshold.json`` next to the weights, otherwise from
``classification.decision_threshold`` in the config.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import tensorflow as tf

from .cache_builder import build_cache_for_mode
from .common import abs_from_repo, load_config, load_preprocessed_input
from .dataset import infer_sample_shape
from .mat_io import load_mat_var
from .models import MaskedSliceAggregator, PairedComplexEchoGate, SqueezeLastDim, build_memod
from .preprocess import swap_to_nxye


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def _confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, int]:
    yt = (np.asarray(y_true).reshape(-1) > 0.5).astype(np.int32)
    yp = (np.asarray(y_prob).reshape(-1) >= float(thr)).astype(np.int32)
    return {
        "tp": int(np.sum((yp == 1) & (yt == 1))),
        "tn": int(np.sum((yp == 0) & (yt == 0))),
        "fp": int(np.sum((yp == 1) & (yt == 0))),
        "fn": int(np.sum((yp == 0) & (yt == 1))),
    }


def _metrics(conf: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * prec * rec, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1}


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    metric = tf.keras.metrics.AUC(curve="ROC")
    metric.update_state(y_true.astype(np.float32), y_prob.astype(np.float32))
    return float(metric.result().numpy())


def _resolve_weights_file(weights_path: str) -> str:
    p = abs_from_repo(weights_path)
    if not os.path.isdir(p):
        return p
    candidates = [
        "best_weights.h5",
        "latest_weights.h5",
        "final_weights.h5",
        "final.keras",
    ]
    for name in candidates:
        fp = os.path.join(p, name)
        if os.path.isfile(fp):
            print(f"[ME-MoD] --weights is a directory; auto-selected: {fp}")
            return fp
    raise FileNotFoundError(f"No weight/model file found in {p}")


def _load_saved_threshold(weights_path: str) -> Optional[float]:
    resolved = _resolve_weights_file(weights_path)
    run_dir = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)
    path = os.path.join(run_dir, "decision_threshold.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    return float(payload["threshold"]) if "threshold" in payload else None


def _resolve_decision_threshold(config: Dict, weights_path: str, cli_threshold: Optional[float]) -> Tuple[float, str]:
    if cli_threshold is not None:
        return float(cli_threshold), "cli"
    saved = _load_saved_threshold(weights_path)
    if saved is not None:
        return float(saved), "saved_validation"
    return float(config.get("classification", {}).get("decision_threshold", 0.5)), "config"


def _custom_objects():
    return {
        "PairedComplexEchoGate": PairedComplexEchoGate,
        "MaskedSliceAggregator": MaskedSliceAggregator,
        "SqueezeLastDim": SqueezeLastDim,
    }


def _load_model(config: Dict, slice_shape: Tuple[int, int, int], weights_path: str) -> tf.keras.Model:
    resolved = _resolve_weights_file(weights_path)
    if resolved.endswith(".keras"):
        return tf.keras.models.load_model(resolved, custom_objects=_custom_objects())
    model = build_memod(
        slice_shape=slice_shape,
        model_cfg=config.get("model", {}),
        aggregation_cfg=dict(config.get("classification", {}).get("aggregation", {})),
    )
    model.load_weights(resolved)
    return model


def _predict_subject(model: tf.keras.Model, x_subject: np.ndarray) -> Tuple[float, np.ndarray]:
    x_subject = np.asarray(x_subject, dtype=np.float32)
    mask = np.ones((1, int(x_subject.shape[0])), dtype=np.float32)
    outputs = model([x_subject[np.newaxis, ...], mask], training=False)
    p_subject, p_slice = outputs[0], outputs[1]
    return float(np.asarray(p_subject).reshape(-1)[0]), np.asarray(p_slice).reshape(-1)[: x_subject.shape[0]]


def _validate_deploy_shape(model: tf.keras.Model, x_subject: np.ndarray, input_mat: str) -> None:
    """Fail early with the offending file when preprocessing axes are wrong."""
    model_shape = model.input_shape
    if isinstance(model_shape, list):
        model_shape = model_shape[0]
    expected = tuple(model_shape[2:])
    actual = tuple(int(v) for v in x_subject.shape[1:])
    mismatch = len(expected) != len(actual) or any(
        dim is not None and int(dim) != value for dim, value in zip(expected, actual)
    )
    if mismatch:
        raise ValueError(
            f"Preprocessed input {input_mat} has per-slice shape {actual}, but the "
            f"trained model expects {expected}. Check the MATLAB variable and its "
            "axis order; expected source layout is (height, width, slices, echoes)."
        )


def _slice_offset(config: Dict, original_num_slices: int) -> int:
    slice_rng = config.get("setting", {}).get("slice_rng", None)
    if slice_rng is None:
        return 0
    if isinstance(slice_rng, (list, tuple)) and len(slice_rng) == 2:
        return int(slice_rng[0])
    if isinstance(slice_rng, dict):
        if str(original_num_slices) in slice_rng:
            return int(slice_rng[str(original_num_slices)][0])
        if int(original_num_slices) in slice_rng:
            return int(slice_rng[int(original_num_slices)][0])
    return 0


def _subject_from_input_path(input_mat: str) -> str:
    folder = os.path.basename(os.path.dirname(input_mat))
    stem = os.path.splitext(os.path.basename(input_mat))[0]
    if folder:
        return folder
    return stem


def _print_deploy_summary(row: Dict) -> None:
    subject = str(row.get("subject", "unknown"))
    pred_name = str(row.get("pred_label_name", "unknown")).upper()
    p_subject = float(row.get("p_subject", 0.0))
    threshold = float(row.get("threshold", 0.5))
    print(f"[ME-MoD] {subject}: {pred_name} (p_subject={p_subject:.4f}, threshold={threshold:.4f})")


def _keep_min_consecutive(indices: np.ndarray, min_consecutive: int) -> np.ndarray:
    if indices.size == 0 or int(min_consecutive) <= 1:
        return indices
    values = np.sort(np.unique(indices.astype(np.int32)))
    keep = []
    start = 0
    for i in range(1, values.size + 1):
        at_end = i == values.size
        if at_end or values[i] != values[i - 1] + 1:
            run = values[start:i]
            if run.size >= int(min_consecutive):
                keep.append(run)
            start = i
    if not keep:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(keep, axis=0)


def _localize(p_slice: np.ndarray, loc_cfg: Dict, top_k: int, offset: int) -> Dict:
    """Report slices whose motion probability exceeds a threshold (descriptive only)."""
    thr = float(loc_cfg.get("threshold", 0.5))
    max_fraction = loc_cfg.get("max_fraction", 0.12)
    max_count = loc_cfg.get("max_count", 12)
    min_consecutive = int(loc_cfg.get("min_consecutive", 3))
    scores = np.asarray(p_slice, dtype=np.float32).reshape(-1)
    candidates = np.where(scores >= thr)[0].astype(np.int32)
    candidates = _keep_min_consecutive(candidates, min_consecutive)
    max_allowed = len(scores)
    if max_fraction is not None:
        max_allowed = min(max_allowed, max(0, int(math.ceil(float(max_fraction) * float(len(scores))))))
    if max_count is not None:
        max_allowed = min(max_allowed, int(max_count))
    if candidates.size > max_allowed:
        order = np.argsort(scores[candidates])[::-1]
        candidates = candidates[order[:max_allowed]]
    top_indices = np.argsort(scores)[-int(top_k) :][::-1].tolist() if top_k > 0 else []
    return {
        "enabled": bool(loc_cfg.get("enable", True)),
        "threshold": thr,
        "max_fraction": max_fraction,
        "max_count": max_count,
        "min_consecutive": min_consecutive,
        "selected_indices": [int(i) for i in candidates.tolist()],
        "selected_indices_original": [int(offset) + int(i) for i in candidates.tolist()],
        "selected_scores": [float(scores[i]) for i in candidates.tolist()],
        "top_indices": [int(i) for i in top_indices],
        "top_indices_original": [int(offset) + int(i) for i in top_indices],
        "top_scores": [float(scores[i]) for i in top_indices],
        "n_selected": int(candidates.size),
    }


def _deploy_single(
    config: Dict,
    model: tf.keras.Model,
    input_mat: str,
    threshold: float,
    threshold_source: str,
    top_slices: int,
    var_candidates: Optional[List[str]] = None,
) -> Dict:
    input_mat_abs = abs_from_repo(input_mat)
    original_vol = load_mat_var(
        input_mat_abs,
        preferred_keys=tuple(var_candidates) if var_candidates else ("ima_comb", "ima_comb1"),
    )
    original_num_slices = int(swap_to_nxye(original_vol).shape[0])
    offset = _slice_offset(config, original_num_slices)
    x = load_preprocessed_input(input_mat_abs, config, var_candidates=var_candidates)
    _validate_deploy_shape(model, x, input_mat_abs)
    p_subject, p_slice = _predict_subject(model, x)
    pred = int(p_subject >= threshold)
    loc_cfg = dict(config.get("classification", {}).get("localization", {}))
    loc_info = _localize(p_slice, loc_cfg=loc_cfg, top_k=int(top_slices), offset=offset) if bool(loc_cfg.get("enable", True)) else None
    all_indices = list(range(int(x.shape[0])))
    return {
        "mode": "deploy",
        "model": "ME-MoD",
        "config": None,
        "weights": None,
        "input_mat": input_mat_abs,
        "subject": _subject_from_input_path(input_mat_abs),
        "original_n_slices": original_num_slices,
        "n_slices": int(x.shape[0]),
        "slice_offset": int(offset),
        "slice_rng": config.get("setting", {}).get("slice_rng", None),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "p_subject": float(p_subject),
        "pred_label": pred,
        "pred_label_name": "motion" if pred == 1 else "motion_free",
        "localization": loc_info,
        "all_slices": {
            "indices": [int(i) for i in all_indices],
            "indices_original": [int(offset) + int(i) for i in all_indices],
            "scores": [float(v) for v in np.asarray(p_slice, dtype=np.float32).reshape(-1)[: int(x.shape[0])].tolist()],
        },
        "top_slices": {
            "indices": (loc_info or {}).get("top_indices", []),
            "indices_original": (loc_info or {}).get("top_indices_original", []),
            "scores": (loc_info or {}).get("top_scores", []),
        },
    }


def _discover_input_mats(input_root: str, mat_glob: str) -> List[str]:
    root = abs_from_repo(input_root)
    files: List[str] = []
    files.extend(glob(os.path.join(root, mat_glob)))
    files.extend(glob(os.path.join(root, "*", mat_glob)))
    deduped = sorted({os.path.abspath(p) for p in files if os.path.isfile(p)})
    return deduped


def run_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    threshold, threshold_source = _resolve_decision_threshold(config, args.weights, args.threshold)
    cache_files = build_cache_for_mode(config, mode=args.split, cache_root=args.cache_root, rebuild_cache=args.rebuild_cache)
    slice_shape = infer_sample_shape(cache_files)
    model = _load_model(config, slice_shape, args.weights)

    y_true_list: List[float] = []
    y_prob_list: List[float] = []
    rows: List[Dict] = []
    for cache_file in cache_files:
        with h5py.File(cache_file, "r") as f:
            x = np.asarray(f["x"][:], dtype=np.float32)
            y_subject = float(np.asarray(f["y_subject"][:]).reshape(-1)[0])
            subject = str(f.attrs.get("subject", os.path.basename(cache_file)))
            source_kind = str(f.attrs.get("source_kind", "unknown"))
            source_name = str(f.attrs.get("source_name", source_kind))
        p_subject, p_slice = _predict_subject(model, x)
        pred = int(p_subject >= threshold)
        rows.append(
            {
                "subject": subject,
                "source_kind": source_kind,
                "source_name": source_name,
                "true_label": int(y_subject > 0.5),
                "pred_label": pred,
                "p_subject": float(p_subject),
                "n_slices": int(x.shape[0]),
                "top_slices": [int(i) for i in np.argsort(p_slice)[-int(args.top_slices) :][::-1].tolist()],
            }
        )
        y_true_list.append(y_subject)
        y_prob_list.append(p_subject)

    y_true = np.asarray(y_true_list, dtype=np.float32)
    y_prob = np.asarray(y_prob_list, dtype=np.float32)
    conf = _confusion(y_true, y_prob, threshold)
    report = {
        "mode": "eval",
        "model": "ME-MoD",
        "config": abs_from_repo(args.config),
        "weights": abs_from_repo(args.weights),
        "split": args.split,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "metrics": {**_metrics(conf), **conf, "auc": _auc(y_true, y_prob), "n_subjects": int(y_true.size)},
        "rows": rows if args.include_rows else None,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "rows" or args.include_rows}, indent=2))
    if args.output_json:
        with open(abs_from_repo(args.output_json), "w") as f:
            json.dump(report, f, indent=2)
    return 0


def run_deploy(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    threshold, threshold_source = _resolve_decision_threshold(config, args.weights, args.threshold)
    var_candidates = [v.strip() for v in args.var_candidates.split(",") if v.strip()] if args.var_candidates else None
    x = load_preprocessed_input(args.input_mat, config, var_candidates=var_candidates)
    slice_shape = tuple(int(v) for v in x.shape[1:])
    model = _load_model(config, slice_shape, args.weights)
    out = _deploy_single(
        config=config,
        model=model,
        input_mat=args.input_mat,
        threshold=threshold,
        threshold_source=threshold_source,
        top_slices=int(args.top_slices),
        var_candidates=var_candidates,
    )
    out["config"] = abs_from_repo(args.config)
    out["weights"] = abs_from_repo(args.weights)
    _print_deploy_summary(out)
    if args.output_json:
        with open(abs_from_repo(args.output_json), "w") as f:
            json.dump(out, f, indent=2)
        print(f"[ME-MoD] wrote report: {abs_from_repo(args.output_json)}")
    return 0


def run_deploy_folder(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    threshold, threshold_source = _resolve_decision_threshold(config, args.weights, args.threshold)
    var_candidates = [v.strip() for v in args.var_candidates.split(",") if v.strip()] if args.var_candidates else None
    input_mats = _discover_input_mats(args.input_root, args.mat_glob)
    if not input_mats:
        raise FileNotFoundError(
            f"No input MAT files found under {abs_from_repo(args.input_root)} "
            f"matching '{args.mat_glob}' at root or one level below"
        )
    first_x = load_preprocessed_input(input_mats[0], config, var_candidates=var_candidates)
    slice_shape = tuple(int(v) for v in first_x.shape[1:])
    model = _load_model(config, slice_shape, args.weights)

    rows: List[Dict] = []
    for input_mat in input_mats:
        row = _deploy_single(
            config=config,
            model=model,
            input_mat=input_mat,
            threshold=threshold,
            threshold_source=threshold_source,
            top_slices=int(args.top_slices),
            var_candidates=var_candidates,
        )
        rows.append(row)
        _print_deploy_summary(row)

    pred_arr = np.asarray([int(r["pred_label"]) for r in rows], dtype=np.int32)
    report = {
        "mode": "deploy_folder",
        "model": "ME-MoD",
        "config": abs_from_repo(args.config),
        "weights": abs_from_repo(args.weights),
        "input_root": abs_from_repo(args.input_root),
        "mat_glob": str(args.mat_glob),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "summary": {
            "n_subjects": int(len(rows)),
            "n_motion": int(np.sum(pred_arr == 1)),
            "n_motion_free": int(np.sum(pred_arr == 0)),
        },
        "rows": rows,
    }
    print(
        "[ME-MoD] folder summary: "
        f"{report['summary']['n_subjects']} volumes, "
        f"{report['summary']['n_motion']} motion, "
        f"{report['summary']['n_motion_free']} motion_free"
    )
    if args.output_json:
        with open(abs_from_repo(args.output_json), "w") as f:
            json.dump(report, f, indent=2)
        print(f"[ME-MoD] wrote report: {abs_from_repo(args.output_json)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate or deploy the ME-MoD motion detector.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="Evaluate a labelled split defined in the config.")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--weights", required=True)
    p_eval.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    p_eval.add_argument("--cache_root", default="cache/memod")
    p_eval.add_argument("--rebuild_cache", action="store_true")
    p_eval.add_argument("--threshold", type=float, default=None)
    p_eval.add_argument("--top_slices", type=int, default=10)
    p_eval.add_argument("--include_rows", action="store_true")
    p_eval.add_argument("--output_json", default=None)
    p_eval.set_defaults(func=run_eval)

    p_dep = sub.add_parser("deploy", help="Classify one MAT volume.")
    p_dep.add_argument("--config", required=True)
    p_dep.add_argument("--weights", required=True)
    p_dep.add_argument("--input_mat", required=True)
    p_dep.add_argument("--threshold", type=float, default=None)
    p_dep.add_argument("--top_slices", type=int, default=10, help="Number of highest-scoring slices to report.")
    p_dep.add_argument("--var_candidates", default=None, help="Comma-separated MAT variable names, e.g. ima_comb,ima_comb1")
    p_dep.add_argument("--output_json", default=None)
    p_dep.set_defaults(func=run_deploy)

    p_dep_folder = sub.add_parser("deploy_folder", help="Classify every MAT volume in a folder.")
    p_dep_folder.add_argument("--config", required=True)
    p_dep_folder.add_argument("--weights", required=True)
    p_dep_folder.add_argument("--input_root", required=True)
    p_dep_folder.add_argument("--mat_glob", default="ima_comb_*.mat")
    p_dep_folder.add_argument("--threshold", type=float, default=None)
    p_dep_folder.add_argument("--top_slices", type=int, default=10)
    p_dep_folder.add_argument("--var_candidates", default=None)
    p_dep_folder.add_argument("--output_json", default=None)
    p_dep_folder.set_defaults(func=run_deploy_folder)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
