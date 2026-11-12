"""Subprocess interface to the ME-MoD detector and the ME-MoC corrector.

Each detection and correction runs in a fresh Python process
(``python -m memod.test deploy`` / ``python -m memoc.test --motion_only``),
so TensorFlow memory is released between passes and every pass is logged
with its exact command and configuration.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MEMOD_CONFIG = REPO_ROOT / "weights" / "memod" / "config.json"
DEFAULT_MEMOD_WEIGHTS = REPO_ROOT / "weights" / "memod" / "best_weights.h5"
DEFAULT_MEMOC_CONFIG = REPO_ROOT / "weights" / "memoc" / "config.json"
DEFAULT_MEMOC_WEIGHTS = REPO_ROOT / "weights" / "memoc" / "best-psnr.h5"


def python_env(disable_onednn: bool = False) -> Dict[str, str]:
    env = os.environ.copy()
    if disable_onednn:
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    return env


def run_detection(
    *,
    memod_config: str,
    memod_weights: str,
    input_mat: str,
    threshold: Optional[float] = None,
    top_slices: int = 12,
    var_candidates: Optional[str] = "ima_comb,ima_comb1",
    disable_onednn: bool = False,
) -> Dict:
    """Run ME-MoD on one MAT volume and return its JSON report.

    The report contains ``p_subject`` (volume motion probability),
    ``pred_label`` (1 = motion, 0 = motion-free), ``threshold`` and per-slice scores.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix="_memod_report.json", delete=False) as tmp:
        report_path = tmp.name

    cmd: List[str] = [
        sys.executable, "-m", "memod.test", "deploy",
        "--config", str(memod_config),
        "--weights", str(memod_weights),
        "--input_mat", str(input_mat),
        "--top_slices", str(int(top_slices)),
        "--output_json", report_path,
    ]
    if threshold is not None:
        cmd += ["--threshold", str(float(threshold))]
    if var_candidates:
        cmd += ["--var_candidates", var_candidates]

    print("[MDCC] running ME-MoD motion detection...", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=python_env(disable_onednn))

    with open(report_path, "r") as f:
        report = json.load(f)
    try:
        os.remove(report_path)
    except OSError:
        pass
    return report


def stage_iteration_input(*, source_mat: str, iter_root: str, base_subject: str, iteration: int) -> Tuple[str, str, str]:
    """Copy a volume to ``<iter_root>/<subject>_iter<N>/ima_comb_<subject>_iter<N>.mat``.

    Returns (staged MAT path, staged subject id, motion root) in the layout
    expected by ``memoc.test``.
    """
    iter_subject = f"{base_subject}_iter{int(iteration)}"
    iter_subject_dir = os.path.join(iter_root, iter_subject)
    os.makedirs(iter_subject_dir, exist_ok=True)
    staged_mat = os.path.join(iter_subject_dir, f"ima_comb_{iter_subject}.mat")
    shutil.copy2(source_mat, staged_mat)
    return staged_mat, iter_subject, iter_root


def _snapshot_summaries(output_dir: str) -> set:
    return set(glob(os.path.join(output_dir, "*", "test_summary.json")))


def _newest_summary(output_dir: str, before: set) -> Optional[str]:
    after = _snapshot_summaries(output_dir)
    candidates = sorted(after - before)
    if not candidates:
        candidates = sorted(after)
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def run_correction(
    *,
    memoc_config: str,
    memoc_weights: str,
    subject: str,
    motion_root: str,
    output_dir: str,
    batch_size: int = 1,
    no_slice_crop: bool = False,
    restore_native_width: bool = True,
    mat_key: str = "ima_comb",
    disable_onednn: bool = False,
) -> Dict:
    """Run ME-MoC (motion-only inference) on ``<motion_root>/<subject>/...``.

    The corrected volume is saved at the input intensity scale and geometry.
    Returns the test summary, the corrected MAT path and the command.
    """
    os.makedirs(output_dir, exist_ok=True)
    before = _snapshot_summaries(output_dir)
    cmd: List[str] = [
        sys.executable, "-m", "memoc.test",
        "--config", str(memoc_config),
        "--weights", str(memoc_weights),
        "--subjects", subject,
        "--motion_only",
        "--motion_root", str(motion_root),
        "--batch_size", str(int(batch_size)),
        "--output_dir", str(output_dir),
        "--mat_key", mat_key,
    ]
    if no_slice_crop:
        cmd.append("--no_slice_crop")
    if restore_native_width:
        cmd.append("--restore_native_width")

    print("[MDCC] running ME-MoC motion correction...", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=python_env(disable_onednn))

    summary_path = _newest_summary(output_dir, before)
    summary = None
    corrected_mat = None
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        per_subject = summary.get("per_subject", [])
        if per_subject:
            corrected_mat = per_subject[0].get("corrected_mat")
    return {
        "summary_path": summary_path,
        "summary": summary,
        "corrected_mat": corrected_mat,
        "command": cmd,
    }


def correct_volume(
    *,
    source_mat: str,
    subject: str,
    iteration: int,
    work_dir: Path,
    memoc_config: str,
    memoc_weights: str,
    batch_size: int = 1,
    no_slice_crop: bool = False,
    disable_onednn: bool = False,
) -> Tuple[Path, Dict]:
    """One ME-MoC correction pass in an isolated, new ``work_dir``.

    The input is staged into ``work_dir/input``, a pass-specific copy of the
    ME-MoC config is written to ``work_dir/correction_config.json`` and the
    output is written to ``work_dir/outputs``. Returns the corrected MAT path
    and the correction record.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    staged, staged_subject, motion_root = stage_iteration_input(
        source_mat=str(source_mat), iter_root=str(work_dir / "input"),
        base_subject=subject, iteration=iteration)
    cfg = json.loads(Path(memoc_config).read_text(encoding="utf-8"))
    cfg.setdefault("data", {}).update({
        "motion_root": motion_root,
        "motion_label": "",
        "test_motion_label": "",
        "test_motion_labels": [""],
        "test_subjects": [staged_subject],
        "motion_mat_name": Path(staged).name,
        "motion_var_candidates": ["ima_comb"],
    })
    cfg_path = work_dir / "correction_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    result = run_correction(
        memoc_config=str(cfg_path), memoc_weights=str(memoc_weights),
        subject=staged_subject, motion_root=motion_root, output_dir=str(work_dir / "outputs"),
        batch_size=batch_size, no_slice_crop=no_slice_crop,
        restore_native_width=True, mat_key="ima_comb", disable_onednn=disable_onednn)
    summary = result.get("summary") or {}
    entries = summary.get("per_subject", [])
    if len(entries) != 1 or entries[0].get("subject") != staged_subject:
        raise ValueError("Correction summary must contain exactly the requested subject")
    corrected = Path(result.get("corrected_mat") or "__missing__")
    if not corrected.is_absolute():
        corrected = REPO_ROOT / corrected
    if not corrected.is_file() or not corrected.resolve().is_relative_to(work_dir.resolve()):
        raise ValueError("Missing correction output or output outside this pass directory")
    result["persistent_config"] = str(cfg_path)
    return corrected, result
