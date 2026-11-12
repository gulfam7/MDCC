"""Cohort evaluation of the MDCC detect -> correct -> re-detect loop.

For every volume in ``--input-dir`` the script runs:
  * the gated policy: ME-MoD decides after every pass whether another ME-MoC
    pass is applied, up to ``--max-iters`` correction passes;
  * a fixed-pass stress test: ``--stress-iters`` ME-MoC passes are applied
    regardless of the detector, to measure the effect of repeated correction.

With an aligned motion-free reference (``--reference-dir``) it reports SSIM,
NRMSE, PSNR, bias and derivative errors per pass; without one it reports
iteration counts and image-change/sharpness diagnostics. See README.md for
example commands. ME-MoD and ME-MoC run in subprocesses (pipeline/bridge.py).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage, stats
from scipy.io import loadmat, savemat
from skimage.metrics import structural_similarity

from . import bridge


ROOT = Path(__file__).resolve().parent.parent
METRICS = ("ssim", "nrmse_pct", "psnr_db", "bias_pct", "gradient_energy",
           "laplacian_variance", "gradient_error", "laplacian_error",
           "change_from_input_pct", "change_from_previous_pct")
LOWER_BETTER = {"nrmse_pct", "gradient_error", "laplacian_error"}
QUALITY = LOWER_BETTER | {"ssim", "psnr_db"}
# Metric definitions (robust scaling, foreground support, NRMSE, PSNR, SSIM)
# match the cohort-level reconstruction metrics reported in the paper.
EPS = 1e-8


def robust_scale_factor(pred_mag, ref_mag, mask):
    """Per-slice/per-echo least-squares gain that aligns |pred| to |ref| inside the mask."""
    valid = (np.asarray(mask, dtype=bool) & np.isfinite(pred_mag) & np.isfinite(ref_mag)
             & (pred_mag > EPS) & (ref_mag > EPS))
    if int(np.sum(valid)) < 32:
        return 1.0
    pred = np.asarray(pred_mag[valid], dtype=np.float64)
    ref = np.asarray(ref_mag[valid], dtype=np.float64)
    denominator = float(np.dot(pred, pred))
    if denominator > EPS:
        scale = float(np.dot(pred, ref) / denominator)
    else:
        scale = float(np.percentile(ref, 95) / (np.percentile(pred, 95) + EPS))
    return scale if math.isfinite(scale) and scale > 0 else 1.0


def bbox_2d(mask):
    ys, xs = np.where(mask)
    if not ys.size or not xs.size:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def global_ssim_fallback(a, b, data_range):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a)), float(np.var(b))
    covariance = float(np.mean((a - mu_a) * (b - mu_b)))
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if denominator <= 0:
        return float("nan")
    return float(((2 * mu_a * mu_b + c1) * (2 * covariance + c2)) / denominator)


def cohort_ssim_2d(a, b, data_range):
    """Gaussian-window SSIM (sigma 1.5, window <= 11) on the foreground bounding box."""
    if min(a.shape[:2]) < 7 or data_range <= 0:
        return float("nan")
    win = min(11, int(min(a.shape[:2])))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float("nan")
    try:
        return float(structural_similarity(
            a, b, data_range=float(data_range), gaussian_weights=True,
            sigma=1.5, use_sample_covariance=False, win_size=win,
        ))
    except Exception:
        return global_ssim_fallback(a, b, data_range)


def write_json(path, value):
    # Standards-compliant JSON; unavailable/non-finite metrics become null.
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        if isinstance(v, np.generic):
            return v.item()
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_volume(path, keys, layout):
    """Read MATLAB logical dimensions explicitly, returning complex (S,H,W,E)."""
    import h5py
    if h5py.is_hdf5(str(path)):
        with h5py.File(path, "r") as f:
            key = next((k for k in keys if k in f), None)
            if key is None:
                raise KeyError(f"{path}: none of {keys} present")
            dataset = f[key]
            data = dataset[()]
            # Genuine MATLAB HDF5 arrays store dimensions in reversed order.
            if "MATLAB_class" in dataset.attrs:
                data = data.transpose(tuple(reversed(range(data.ndim))))
    else:
        contents = loadmat(str(path), variable_names=list(keys))
        key = next((k for k in keys if k in contents), None)
        if key is None:
            raise KeyError(f"{path}: none of {keys} present")
        data = contents[key]
    if data.dtype.names:
        if not {"real", "imag"}.issubset(data.dtype.names):
            raise ValueError(f"{path}: unsupported compound complex representation")
        data = data["real"] + 1j * data["imag"]
    if data.ndim == 5 and data.shape[-1] == 2 and not np.iscomplexobj(data):
        data = data[..., 0] + 1j * data[..., 1]
    if data.ndim != 4:
        raise ValueError(f"{path}: expected 4-D multi-echo volume, got {data.shape}")
    if not np.iscomplexobj(data):
        raise ValueError(f"{path}: complex data required for the complex detector/corrector")
    if layout == "hwse":
        data = data.transpose(2, 0, 1, 3)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: NaN/Inf in volume; fix input instead of changing masks per pass")
    return data.astype(np.complex64, copy=False)


def save_volume(path, data):
    savemat(str(path), {"ima_comb": data.transpose(1, 2, 0, 3)})


class Evaluator:
    """Fixed spatial support with selectable, explicitly recorded metric scaling."""
    def __init__(self, original, reference, slice_range, roi, threshold,
                 metric_normalization="robust_scale"):
        if reference is not None and reference.shape != original.shape:
            raise ValueError(f"GT/input shapes differ: {reference.shape} vs {original.shape}; registration/cropping is not automatic")
        self.original = original
        self.reference = reference
        self.anchor = np.abs(reference if reference is not None else original)
        self.shape = original.shape
        self.metric_normalization = metric_normalization
        if metric_normalization not in {"robust_scale", "fixed_reference_peak"}:
            raise ValueError(f"Unsupported metric normalization: {metric_normalization}")
        n, h, w, _ = original.shape
        lo, hi = slice_range or (1, n)
        if not 1 <= lo <= hi <= n:
            raise ValueError(f"Slice range {lo}:{hi} outside 1:{n}")
        x, y, rw, rh = roi or (0, 0, w, h)
        if x < 0 or y < 0 or rw < 7 or rh < 7 or x + rw > w or y + rh > h:
            raise ValueError(f"ROI {(x, y, rw, rh)} invalid for (H,W)={(h,w)}; minimum size 7x7")
        self.selection = (slice(y, y + rh), slice(x, x + rw))
        self.slices = list(range(lo - 1, hi))
        self.masks = {}
        self.full_masks = {}
        for s in self.slices:
            first = self.anchor[s, ..., 0]
            mask = first >= max(threshold * float(first.max()), EPS)
            self.full_masks[s] = mask
            region_mask = mask[self.selection]
            if roi is not None and not region_mask.any():
                region_mask = np.ones((rh, rw), dtype=bool)
            self.masks[s] = region_mask
        if not any(ndimage.binary_erosion(m, iterations=3, structure=np.ones((3, 3))).any() for m in self.masks.values()):
            raise ValueError("No valid evaluation pixels with a 7x7 SSIM neighborhood; revise fixed slice/ROI selection")

    def evaluate(self, current, previous, subject, iteration, phase):
        if current.shape != self.shape or previous.shape != self.shape:
            raise ValueError(f"Correction changed geometry: {current.shape} vs {self.shape}")
        rows = []
        for s in self.slices:
            mask = self.masks[s]
            inner = ndimage.binary_erosion(mask, iterations=1)
            centers = ndimage.binary_erosion(mask, iterations=3, structure=np.ones((3, 3)))
            for echo in range(self.shape[-1]):
                row = {"subject": subject, "iteration": iteration, "phase": phase,
                       "slice": s + 1, "echo": echo + 1, "n_voxels": int(mask.sum()),
                       "n_ssim_centers": int(centers.sum()), "has_reference": self.reference is not None}
                row.update({k: float("nan") for k in METRICS})
                full_a = self.anchor[s, ..., echo].astype(np.float64)
                full_b = np.abs(current[s, ..., echo]).astype(np.float64)
                full_initial = np.abs(self.original[s, ..., echo]).astype(np.float64)
                full_prev = np.abs(previous[s, ..., echo]).astype(np.float64)
                row["metric_scale_factor"] = 1.0
                if self.reference is not None and self.metric_normalization == "robust_scale":
                    scale_mask = self.full_masks[s]
                    row["metric_scale_factor"] = robust_scale_factor(full_b, full_a, scale_mask)
                    full_b = full_b * row["metric_scale_factor"]
                    full_initial = full_initial * robust_scale_factor(full_initial, full_a, scale_mask)
                    full_prev = full_prev * robust_scale_factor(full_prev, full_a, scale_mask)

                a = full_a[self.selection]
                b = full_b[self.selection]
                initial = full_initial[self.selection]
                prev = full_prev[self.selection]
                peak = float(a[mask].max()) if mask.any() else 0.0
                data_range = float(a[mask].max() - a[mask].min()) if mask.any() else 0.0
                descriptive_scale = data_range if self.reference is not None and self.metric_normalization == "robust_scale" else peak
                if descriptive_scale <= 0 or not inner.any():
                    rows.append(row)
                    continue
                a_norm, b_norm = a / descriptive_scale, b / descriptive_scale
                initial_norm, prev_norm = initial / descriptive_scale, prev / descriptive_scale
                gy, gx = np.gradient(b_norm)
                lap = ndimage.laplace(b_norm)
                row["gradient_energy"] = float(np.mean((gx * gx + gy * gy)[inner]))
                row["laplacian_variance"] = float(np.var(lap[inner]))
                row["change_from_input_pct"] = 100 * float(np.sqrt(np.mean((b_norm[mask] - initial_norm[mask]) ** 2)))
                row["change_from_previous_pct"] = 100 * float(np.sqrt(np.mean((b_norm[mask] - prev_norm[mask]) ** 2)))
                if self.reference is not None:
                    residual = b[mask] - a[mask]
                    rmse = float(np.sqrt(np.mean(residual * residual)))
                    denominator = data_range if self.metric_normalization == "robust_scale" else peak
                    row["nrmse_pct"] = 100 * rmse / (denominator + EPS)
                    row["psnr_db"] = float("inf") if rmse <= 0 else float(20 * np.log10(peak / (rmse + EPS)))
                    row["bias_pct"] = 100 * float(np.mean(residual)) / (denominator + EPS)
                    ay, ax = np.gradient(a_norm)
                    row["gradient_error"] = float(np.sqrt(np.mean(((gx - ax) ** 2 + (gy - ay) ** 2)[inner])))
                    row["laplacian_error"] = float(np.sqrt(np.mean((lap[inner] - ndimage.laplace(a_norm)[inner]) ** 2)))
                    if self.metric_normalization == "robust_scale":
                        bbox = bbox_2d(mask)
                        if bbox is not None:
                            y0, y1, x0, x1 = bbox
                            row["ssim"] = cohort_ssim_2d(a[y0:y1, x0:x1], b[y0:y1, x0:x1], max(data_range, EPS))
                    elif centers.any():
                        _, ssim_map = structural_similarity(a_norm, b_norm, data_range=1.0, win_size=7, full=True)
                        row["ssim"] = float(np.mean(ssim_map[centers]))
                rows.append(row)
        return rows


def subject_metrics(rows):
    out = {}
    for key in METRICS:
        values = np.array([r[key] for r in rows], dtype=float)
        valid = values[~np.isnan(values)]
        out[key] = float(np.mean(valid)) if len(valid) else float("nan")
        out[key + "_n_slice_echo"] = len(valid)
    return out


class Backend:
    def __init__(self, args):
        self.args = args

    def detect(self, path):
        a = self.args
        report = bridge.run_detection(
            memod_config=a.memod_config, memod_weights=a.memod_weights,
            input_mat=str(path), threshold=a.threshold, top_slices=a.memod_top_slices,
            var_candidates="ima_comb", disable_onednn=a.disable_onednn)
        if report.get("pred_label") not in (0, 1):
            raise ValueError("Detector did not return an explicit binary pred_label")
        if not np.isfinite(report.get("p_subject", float("nan"))):
            raise ValueError("Detector returned missing/nonfinite p_subject")
        return report

    def correct(self, path, subject, iteration, out):
        a = self.args
        # Every pass has its own input folder, config copy and output directory.
        return bridge.correct_volume(
            source_mat=str(path), subject=subject, iteration=iteration,
            work_dir=out / "backend" / f"pass_{iteration:02d}",
            memoc_config=a.memoc_config, memoc_weights=a.memoc_weights,
            batch_size=a.correction_batch_size, no_slice_crop=a.no_slice_crop,
            disable_onednn=a.disable_onednn)


def evaluation_slices(args, num_slices):
    """Default to corrected slices so untouched slices do not dilute effects."""
    if args.slice_range:
        return args.slice_range
    if args.no_slice_crop:
        return [1, num_slices]
    config = json.loads(Path(args.memoc_config).read_text(encoding="utf-8"))
    crop = config.get("setting", {}).get("slice_rng")
    if isinstance(crop, dict):
        crop = crop[str(num_slices)]
    if crop is None:
        return [1, num_slices]
    if len(crop) != 2 or not 0 <= int(crop[0]) < int(crop[1]):
        raise ValueError(f"Invalid correction config slice_rng={crop} for {num_slices} slices")
    start, end = int(crop[0]), min(int(crop[1]), num_slices)
    if start >= end:
        raise ValueError(f"Correction config slice_rng={crop} does not overlap the {num_slices}-slice volume")
    if int(crop[1]) > num_slices:
        print(f"[evaluation] Config slice_rng={crop} clipped to [{start}, {end}] for {num_slices} slices; metric slices {start + 1}-{end} (1-based).", flush=True)
    return [start + 1, end]


def preview(out, evaluator, stages):
    """Fixed subject slice, echo indices and intensity/error ranges for every pass."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = evaluator.slices[len(evaluator.slices) // 2]
    echoes = sorted({0, evaluator.shape[-1] // 2, evaluator.shape[-1] - 1})
    panels = []
    if evaluator.reference is not None:
        panels.append(("Reference", np.abs(evaluator.reference[s][evaluator.selection])))
    for stage in stages:
        volume = read_volume(stage["mat_path"], ["ima_comb"], "hwse")
        panels.append((f"Pass {stage['iteration']} ({stage['phase']})", np.abs(volume[s][evaluator.selection])))
    anchor = evaluator.anchor[s][evaluator.selection]
    fig, axes = plt.subplots(len(echoes), len(panels), squeeze=False,
                             figsize=(3 * len(panels), 3 * len(echoes)))
    for row, echo in enumerate(echoes):
        peak = float(anchor[..., echo].max())
        for col, (label, data) in enumerate(panels):
            axes[row, col].imshow(data[..., echo], cmap="gray", vmin=0, vmax=peak)
            axes[row, col].set_title(f"{label}\nSlice {s+1}, echo {echo+1}", fontsize=9)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out / "iteration_magnitudes.png", dpi=140)
    plt.close(fig)
    if evaluator.reference is not None:
        errors = [(label, np.abs(data - anchor)) for label, data in panels[1:]]
        fig, axes = plt.subplots(len(echoes), len(errors), squeeze=False,
                                 figsize=(3 * len(errors), 3 * len(echoes)))
        for row, echo in enumerate(echoes):
            peak = max(float(data[..., echo].max()) for _, data in errors)
            for col, (label, data) in enumerate(errors):
                im = axes[row, col].imshow(data[..., echo], cmap="magma", vmin=0, vmax=peak or 1.)
                axes[row, col].set_title(f"{label}\nAbsolute magnitude error, echo {echo+1}", fontsize=9)
                axes[row, col].axis("off")
            fig.colorbar(im, ax=list(axes[row]), shrink=.6)
        fig.savefig(out / "iteration_absolute_errors.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def run_case(case, args, backend, out):
    out.mkdir(parents=True, exist_ok=False)
    original = read_volume(case["input"], args.mat_keys.split(","), args.layout)
    reference = read_volume(case["reference"], args.reference_keys.split(","), args.reference_layout) if case["reference"] else None
    selected_slices = evaluation_slices(args, original.shape[0])
    evaluator = Evaluator(original, reference, selected_slices, args.roi, args.mask_threshold,
                          args.metric_normalization)
    current = original
    previous = original
    stages, all_metrics = [], []
    stop_iteration = None
    stop_reason = None
    final_path = None
    limit = max(args.max_iters, args.stress_iters)
    started = time.perf_counter()
    report = {"subject": case["subject"], "input": case["input"], "reference": case["reference"],
              "cohort": args.cohort_label, "status": "running", "stages": stages,
              "mask_source": "reference_echo1" if reference is not None else "input_echo1",
              "shape_shwe": list(original.shape), "evaluation_slices": selected_slices,
              "evaluation_roi": args.roi, "mask_threshold": args.mask_threshold,
              "metric_normalization": args.metric_normalization}
    write_json(out / "report.json", report)
    for iteration in range(limit + 1):
        path = out / f"iteration_{iteration:02d}.mat"
        correction = None
        correction_seconds = 0.0
        phase = "initial" if iteration == 0 else ("gated" if stop_iteration is None else "stress")
        if iteration:
            previous = current
            tick = time.perf_counter()
            corrected_path, correction = backend.correct(stages[-1]["mat_path"], case["subject"], iteration, out)
            correction_seconds = time.perf_counter() - tick
            current = read_volume(corrected_path, ["ima_comb"], "hwse")
        if current.shape != original.shape:
            raise ValueError("Output geometry changed; metrics and redetection would be invalid")
        save_volume(path, current)
        tick = time.perf_counter()
        detection = backend.detect(path)
        detection_seconds = time.perf_counter() - tick
        if detection.get("pred_label") not in (0, 1):
            raise ValueError("Missing/invalid detector label")
        if stop_iteration is None:
            if detection["pred_label"] == 0:
                stop_iteration, stop_reason = iteration, "gate_motion_free"
            elif iteration >= args.max_iters:
                stop_iteration, stop_reason = iteration, "cap_still_motion_positive"
            if stop_iteration is not None:
                final_path = out / "final_gated.mat"
                shutil.copy2(path, final_path)
        rows = evaluator.evaluate(current, previous, case["subject"], iteration, phase)
        all_metrics.extend(rows)
        stage = {"subject": case["subject"], "iteration": iteration, "phase": phase,
                 "mat_path": str(path), "p_subject": detection.get("p_subject"),
                 "pred_label": detection["pred_label"], "threshold": detection.get("threshold"),
                 "detection_seconds": detection_seconds, "correction_seconds": correction_seconds,
                 "detection": detection, "correction": correction, **subject_metrics(rows)}
        stages.append(stage)
        report.update({"gated_corrections": stop_iteration, "stop_reason": stop_reason,
                       "final_gated_mat": str(final_path) if final_path else None})
        write_json(out / "report.json", report)
        write_csv(out / "metrics_per_slice_echo.csv", all_metrics)
        print(f"[{case['subject']}] pass={iteration} phase={phase} p={detection.get('p_subject')} label={detection['pred_label']}", flush=True)
        if stop_iteration is not None and iteration >= args.stress_iters:
            break
    report.update({"status": "complete", "total_seconds": time.perf_counter() - started,
                   "stress_corrections": sum(s["phase"] == "stress" for s in stages),
                   "detector_calls": len(stages),
                   "final_pred_label": stages[stop_iteration]["pred_label"]})
    write_json(out / "report.json", report)
    if not args.no_plots:
        preview(out, evaluator, stages)
    return report


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    v = values[np.isfinite(values)]
    n = len(v)
    mean = float(np.mean(v)) if n else float("nan")
    sd = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    half = float(stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "n_nonfinite": len(values) - n, "mean": mean, "sd": sd,
            "ci95_low": mean - half, "ci95_high": mean + half}


def trajectories(reports, args):
    rows = []
    for report in reports:
        if report["status"] != "complete":
            continue
        stages = report["stages"]
        stop = report["gated_corrections"]
        for iteration in range(args.max_iters + 1):
            source = min(iteration, stop)
            row = {k: v for k, v in stages[source].items() if k not in ("detection", "correction")}
            if iteration > stop:
                row["change_from_previous_pct"] = 0.0
            rows.append({**row, "trajectory": "gated_policy", "iteration": iteration,
                         "source_iteration": source, "carried_forward": iteration > stop})
        if args.stress_iters:
            for stage in stages[:args.stress_iters + 1]:
                rows.append({**{k: v for k, v in stage.items() if k not in ("detection", "correction")},
                             "trajectory": "fixed_pass_stress", "source_iteration": stage["iteration"],
                             "carried_forward": False})
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["trajectory"], row["iteration"])].append(row)
    summaries = []
    pairs = []
    lookup = {(r["trajectory"], r["subject"], r["iteration"]): r for r in rows}
    for (trajectory, iteration), group in sorted(groups.items()):
        for metric in METRICS:
            summaries.append({"trajectory": trajectory, "iteration": iteration, "metric": metric,
                              **finite_summary([r[metric] for r in group])})
            if iteration == 0:
                continue
            comparisons = {"input": 0, "previous": iteration - 1}
            if iteration > 1:
                comparisons["pass_1"] = 1
            for comparison, earlier in comparisons.items():
                deltas = []
                for row in group:
                    old = lookup.get((trajectory, row["subject"], earlier))
                    if old and np.isfinite(row[metric]) and np.isfinite(old[metric]):
                        deltas.append(row[metric] - old[metric])
                d = np.asarray(deltas)
                # Exact zero differences are valid ties, not missing subjects.
                p = (1.0 if not np.any(d) else float(stats.wilcoxon(d, alternative="two-sided").pvalue)) if len(d) >= 2 else float("nan")
                direction = "lower" if metric in LOWER_BETTER else ("higher" if metric in QUALITY else "descriptive_only")
                worse = int(np.sum(d > 0 if direction == "lower" else d < 0)) if metric in QUALITY else ""
                pairs.append({"trajectory": trajectory, "iteration": iteration, "comparison": comparison,
                    "reference_iteration": earlier, "metric": metric, "difference_definition": "current - reference",
                    "direction_better": direction, "n_numerically_worse": worse,
                    "p_raw": p, **finite_summary(d)})
    families = defaultdict(list)
    for row in pairs:
        if np.isfinite(row["p_raw"]):
            families[(row["trajectory"], row["metric"], row["comparison"])].append(row)
    for family in families.values():
        ordered = sorted(family, key=lambda r: r["p_raw"])
        running = 0.0
        for rank, row in enumerate(ordered):
            running = max(running, min(1., (len(ordered) - rank) * row["p_raw"]))
            row["p_holm_across_iterations"] = running
    return summaries, pairs


def plot_results(out, distributions, summaries, has_reference=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    passes = sorted({r["correction_passes"] for r in distributions})
    bottom = np.zeros(len(passes))
    for reason in ("gate_motion_free", "cap_still_motion_positive"):
        counts = [sum(r["n"] for r in distributions if r["correction_passes"] == k and r["stop_reason"] == reason) for k in passes]
        ax.bar(passes, counts, bottom=bottom, label=reason.replace("_", " "))
        bottom += counts
    ax.set(xlabel="Number of gated correction passes", ylabel="Volumes", xticks=passes)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"iteration_counts.{ext}", dpi=180)
    plt.close(fig)
    shown = ("ssim", "nrmse_pct", "psnr_db", "gradient_energy", "laplacian_variance", "gradient_error")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.flat, shown):
        plotted = False
        for trajectory in ("gated_policy", "fixed_pass_stress"):
            selected = [r for r in summaries if r["metric"] == metric and r["trajectory"] == trajectory and r["n"]]
            if selected:
                plotted = True
                x = [r["iteration"] for r in selected]
                ax.plot(x, [r["mean"] for r in selected], "o-", label=trajectory.replace("_", " "))
                ax.fill_between(x, [r["ci95_low"] for r in selected], [r["ci95_high"] for r in selected], alpha=.15)
        ax.set(title=metric.replace("_", " "), xlabel="Correction pass (0 = input)")
        if not plotted:
            message = "Unavailable: no aligned reference\nSupply --reference-dir" if not has_reference and metric in QUALITY else "No finite measurements available"
            ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"metrics_by_iteration.{ext}", dpi=180)
    plt.close(fig)


def export_cohort(reports, args, out):
    complete = [r for r in reports if r["status"] == "complete"]
    subjects = [{k: v for k, v in r.items() if k != "stages"} for r in reports]
    write_csv(out / "subjects.csv", subjects)
    stages = [{k: v for k, v in s.items() if k not in ("detection", "correction")} for r in complete for s in r["stages"]]
    write_csv(out / "observed_iterations.csv", stages)
    counts = Counter((r["gated_corrections"], r["stop_reason"]) for r in complete)
    distribution = [{"correction_passes": k, "stop_reason": reason, "n": n,
                     "denominator_completed": len(complete), "percent": 100 * n / len(complete)}
                    for (k, reason), n in sorted(counts.items())]
    write_csv(out / "iteration_distribution.csv", distribution)
    rows = trajectories(reports, args)
    write_csv(out / "trajectory_subject_metrics.csv", rows)
    summaries, pairs = aggregate(rows)
    write_csv(out / "iteration_metric_summary.csv", summaries)
    write_csv(out / "paired_iteration_tests.csv", pairs)
    lines = ["# Iterative MDCC cohort evaluation", "", f"Cohort: {args.cohort_label}", "",
             f"Completed: {len(complete)} / {len(reports)} attempted volumes; failures: {len(reports)-len(complete)}.", "",
             "Correction counts exclude detector calls and forced stress passes. A capped motion-positive case is not convergence.", "",
             "| Correction passes | Stop reason | Volumes | Percent of completed |", "|---:|---|---:|---:|"]
    lines += [f"| {r['correction_passes']} | {r['stop_reason']} | {r['n']} | {r['percent']:.1f} |" for r in distribution]
    lines += ["", "Gated-policy trajectories carry the accepted image forward after stopping, keeping the same cohort at each pass. Fixed-pass stress trajectories continue correction regardless of the detector. Observed stages preserve the actual phase of every pass.", "",
              "Metrics are per-volume means over the same selected slices/echoes. Dispersion is across volumes. Confidence intervals are t intervals. Paired tests use volume means and Holm correction across iterations within each trajectory/metric/comparison family.", "",
              (("Reference-based reconstruction metrics use per-slice/per-echo least-squares robust_scale alignment; NRMSE is normalised by the reference intensity range and SSIM is computed on the foreground bounding box. "
                if args.metric_normalization == "robust_scale" else
                "Reference-based reconstruction metrics use fixed reference-peak scaling without prediction-specific intensity alignment. ")
               + "The support is thresholded from reference echo 1, or input echo 1 when no reference exists; it is not anatomical segmentation."), "",
              "Without aligned ground truth, SSIM/NRMSE/PSNR, bias and derivative errors are unavailable. Gradient energy and Laplacian variance are descriptive: decreases can reflect denoising, artifact removal or smoothing, and alone do not establish image degradation. Infinite PSNR is retained in subject CSVs but omitted (and counted) in finite statistical summaries.", "",
              f"Evaluation slices (1-based inclusive): {args.slice_range or 'corrector-config crop (all if no crop); see each subject report'}; ROI (x,y,width,height, 0-based): {args.roi or 'full image'}. Correction crop follows the model config unless --no-slice-crop is supplied.", "",
              "| Trajectory | Pass | Metric | N finite | Mean | SD |", "|---|---:|---|---:|---:|---:|"]
    lines += [f"| {r['trajectory']} | {r['iteration']} | {r['metric']} | {r['n']} | {r['mean']:.6g} | {r['sd']:.6g} |" for r in summaries if r["n"]]
    (out / "cohort_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if complete and not args.no_plots:
        plot_results(out, distribution, summaries, has_reference=any(r.get("reference") for r in complete))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    def arg(name, **kwargs):
        names = ["--" + name.replace("_", "-")]
        if "_" in name:
            names.append("--" + name)
        p.add_argument(*names, dest=name, **kwargs)
    arg("input_dir", required=True, help="MAT files directly in this directory and/or in immediate subject subfolders; both layouts are discovered automatically")
    arg("output_dir", required=True, help="New or empty output directory; existing results are never overwritten")
    arg("pattern", default="*.mat")
    arg("recursive", action="store_true", help="Also search deeper than one subject-folder level")
    arg("cohort_label", default="unspecified", help="For example synthetic-light, synthetic-moderate, real-motion, clean")
    arg("subject_regex", help="Override subject extraction with a named (?P<subject>...) group; otherwise use immediate subject-folder name, or filename stem without ima_comb_ for flat files")
    arg("reference_dir", help="Optional aligned ground-truth directory. Missing matches are errors")
    arg("reference_template", default=None, help="Optional exact relative reference path with {subject}, {stem}, {filename}; default auto-matches flat or subject-subfolder references")
    arg("layout", choices=("hwse", "shwe"), default="hwse")
    arg("reference_layout", choices=("hwse", "shwe"), default="hwse")
    arg("mat_keys", default="ima_comb,ima_comb1")
    arg("reference_keys", default="ima_comb,ima_comb1")
    arg("slice_range", nargs=2, type=int, help="Metric slices, 1-based inclusive; defaults to corrector crop; does not change detector/corrector crops")
    arg("roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="Fixed metric rectangle, 0-based x/column and y/row; no separate mask required")
    arg("mask_threshold", type=float, default=.03)
    arg("metric_normalization", choices=("robust_scale", "fixed_reference_peak"), default="robust_scale",
        help="Reference-metric intensity handling; robust_scale matches the previous cohort tables")
    arg("memod_config", default=str(bridge.DEFAULT_MEMOD_CONFIG), help="ME-MoD config (with calibrated threshold)")
    arg("memod_weights", default=str(bridge.DEFAULT_MEMOD_WEIGHTS), help="ME-MoD weights; decision_threshold.json beside it is used")
    arg("memoc_config", default=str(bridge.DEFAULT_MEMOC_CONFIG), help="ME-MoC config used for training")
    arg("memoc_weights", default=str(bridge.DEFAULT_MEMOC_WEIGHTS), help="ME-MoC weights")
    arg("memod_top_slices", type=int, default=12, help="Highest-scoring slices listed in each detection report")
    arg("threshold", type=float, help="Optional fixed decision threshold; otherwise uses the trained detector threshold")
    arg("correction_batch_size", type=int, default=1)
    arg("max_iters", type=int, default=3, help="Maximum gated correction passes; detector also evaluates the final output")
    arg("stress_iters", type=int, default=3, help="Fixed number of correction passes to evaluate even after stopping; 0 disables stress continuation")
    arg("no_slice_crop", action="store_true", help="Correct all slices instead of config crop; model must support the input geometry")
    arg("disable_onednn", action="store_true")
    arg("no_plots", action="store_true")
    arg("dry_run", action="store_true", help="Validate paths and reference matching, save manifest, do not run models")
    return p


def discover(args):
    directory = Path(args.input_dir).resolve()
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    if args.recursive:
        candidates = directory.rglob(args.pattern)
    else:
        # Include both layouts even when a directory contains a mixture of them.
        candidates = list(directory.glob(args.pattern))
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                candidates.extend(child.glob(args.pattern))
    excluded = [Path(getattr(args, name)).resolve() for name in ("output_dir", "reference_dir")
                if getattr(args, name, None)]
    if directory in excluded:
        raise ValueError("Input directory must differ from output and reference directories")
    files = sorted({f.resolve() for f in candidates if f.is_file()
                    and not any(f.resolve().is_relative_to(root) for root in excluded)})
    if not files:
        raise ValueError(f"No inputs matched {directory / args.pattern}")
    cases, seen = [], set()
    for path in files:
        if args.subject_regex:
            match = re.fullmatch(args.subject_regex, path.stem)
            if not match or "subject" not in match.groupdict():
                raise ValueError(f"Filename {path.stem} does not match --subject-regex named subject group")
            subject = match.group("subject")
        elif path.parent.parent == directory:
            subject = path.parent.name
            if path.stem.startswith("ima_comb_"):
                filename_subject = path.stem.removeprefix("ima_comb_")
                if filename_subject != subject and not filename_subject.startswith(subject + "_"):
                    raise ValueError(f"Subject folder {subject!r} conflicts with filename {path.name!r}; correct the folder or use --subject-regex")
        else:
            subject = path.stem.removeprefix("ima_comb_")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", subject):
            raise ValueError(f"Unsafe subject identifier {subject!r}; provide --subject-regex")
        if subject.casefold() in seen:
            raise ValueError(f"Duplicate subject {subject}; run severities/realizations separately to avoid treating repeated volumes as independent")
        seen.add(subject.casefold())
        reference = None
        if args.reference_dir:
            root = Path(args.reference_dir).resolve()
            if args.reference_template:
                reference = (root / args.reference_template.format(subject=subject, stem=path.stem, filename=path.name)).resolve()
                if not reference.is_relative_to(root) or not reference.is_file():
                    raise ValueError(f"Missing/out-of-directory reference for {subject}: {reference}")
            else:
                names = {path.name, f"ima_comb_{subject}.mat"}
                matches = sorted({(folder / name).resolve() for folder in (root, root / subject)
                                  for name in names if (folder / name).is_file()
                                  and (folder / name).resolve().is_relative_to(root)})
                if not matches:
                    raise ValueError(f"Missing reference for {subject}: expected {sorted(names)} in {root} or {root / subject}; use --reference-template for another naming scheme")
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous reference for {subject}: {matches}; use --reference-template to select exactly one")
                reference = matches[0]
            if reference == path:
                raise ValueError("Input and ground truth resolve to the same file; omit reference-dir for an unpaired clean control")
        cases.append({"subject": subject, "input": str(path), "reference": str(reference) if reference else None})
    return cases


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_iters < 1 or args.stress_iters < 0:
        raise ValueError("max-iters must be >=1 and stress-iters >=0")
    if not 0 <= args.mask_threshold < 1 or (args.threshold is not None and not 0 <= args.threshold <= 1):
        raise ValueError("Invalid mask/detection threshold")
    if args.correction_batch_size < 1 or args.memod_top_slices < 1:
        raise ValueError("Batch size and top slices must be positive")
    if not args.no_plots:
        import matplotlib  # fail before inference if the requested plotting dependency is missing
    paths = {}
    for name in ("memod_config", "memod_weights", "memoc_config", "memoc_weights"):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
        setattr(args, name, str(path))
        paths[name] = fingerprint(path)
    for name in ("memod_runner", "memoc_runner"):
        paths[name] = fingerprint(ROOT / name[:5] / "test.py")
    for name in ("memod_config", "memoc_config"):
        json.loads(Path(getattr(args, name)).read_text(encoding="utf-8"))
    cases = discover(args)
    if not args.reference_dir:
        print("[evaluation] No --reference-dir: SSIM, NRMSE, PSNR, bias and reference derivative errors are unavailable. Iteration counts and image-change/sharpness diagnostics will still be saved.", flush=True)
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
             "python": sys.version, "numpy": np.__version__, "models_and_configs": paths,
             "runner": fingerprint(__file__), "bridge": fingerprint(bridge.__file__),
             "inputs": [{**c, "input_fingerprint": fingerprint(c["input"]),
                         "reference_fingerprint": fingerprint(c["reference"]) if c["reference"] else None} for c in cases]}
    write_json(out / "analysis_audit.json", audit)
    write_csv(out / "input_manifest.csv", cases)
    if args.dry_run:
        print(f"Validated {len(cases)} volumes; manifest: {out / 'input_manifest.csv'}")
        return 0
    reports = []
    backend = Backend(args)
    for case in cases:
        subject_out = out / "subjects" / case["subject"]
        try:
            report = run_case(case, args, backend, subject_out)
        except Exception as error:
            report = {**case, "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
            subject_out.mkdir(parents=True, exist_ok=True)
            write_json(subject_out / "failure.json", report)
            print(f"[{case['subject']}] FAILED: {error}", file=sys.stderr, flush=True)
        reports.append(report)
        write_json(out / "progress.json", [{k: v for k, v in r.items() if k != "stages"} for r in reports])
    export_cohort(reports, args, out)
    print(f"Cohort report: {out / 'cohort_summary.md'}")
    return 1 if any(r["status"] != "complete" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
