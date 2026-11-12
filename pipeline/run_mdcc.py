"""Run the MDCC detect -> correct -> re-detect loop on one complex multi-echo volume.

    python -m pipeline.run_mdcc --input_mat path/to/ima_comb_<subj>.mat --output_dir results/mdcc

Loop (pass 0 is the input volume):
  1. ME-MoD classifies the current volume.
  2. motion-free -> stop and return the current volume.
     motion       -> ME-MoC corrects it, and the corrected volume is
                     classified again.
  3. After ``--max_iters`` correction passes the loop stops even if the
     volume is still classified as motion-corrupted.

A volume classified as motion-free at pass 0 is returned unchanged; ME-MoC
is not applied to it. The bundled ME-MoD and ME-MoC checkpoints in
``weights/`` are used unless other files are given.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from . import bridge


def _subject_from_path(path: Path) -> str:
    stem = path.stem
    return stem[len("ima_comb_"):] if stem.startswith("ima_comb_") else stem


def _safe_subject(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "subject"


def run_mdcc(
    *,
    input_mat: Path,
    output_dir: Path,
    subject: Optional[str] = None,
    max_iters: int = 3,
    threshold: Optional[float] = None,
    memod_config: str = str(bridge.DEFAULT_MEMOD_CONFIG),
    memod_weights: str = str(bridge.DEFAULT_MEMOD_WEIGHTS),
    memoc_config: str = str(bridge.DEFAULT_MEMOC_CONFIG),
    memoc_weights: str = str(bridge.DEFAULT_MEMOC_WEIGHTS),
    top_slices: int = 12,
    batch_size: int = 1,
    no_slice_crop: bool = False,
    disable_onednn: bool = False,
    keep_work: bool = False,
) -> Dict:
    """Run the gated loop and return the report (also saved as ``<subject>_mdcc_report.json``)."""
    input_mat = Path(input_mat).resolve()
    if not input_mat.is_file():
        raise FileNotFoundError(input_mat)
    memod_config, memod_weights, memoc_config, memoc_weights = (
        str(Path(p).resolve()) for p in (memod_config, memod_weights, memoc_config, memoc_weights)
    )
    for path in (memod_config, memod_weights, memoc_config, memoc_weights):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    max_iters = max(1, min(5, int(max_iters)))
    subject = _safe_subject(subject or _subject_from_path(input_mat))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / "work" / f"{subject}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    started = time.perf_counter()
    stages: List[Dict] = []
    current = input_mat
    stop_reason = None
    for iteration in range(max_iters + 1):
        correction = None
        if iteration > 0:
            print(f"[MDCC] {subject}: correction pass {iteration}/{max_iters}", flush=True)
            current, correction = bridge.correct_volume(
                source_mat=str(current), subject=subject, iteration=iteration,
                work_dir=work_root / f"pass_{iteration:02d}",
                memoc_config=memoc_config, memoc_weights=memoc_weights,
                batch_size=batch_size, no_slice_crop=no_slice_crop, disable_onednn=disable_onednn)
        detection = bridge.run_detection(
            memod_config=memod_config, memod_weights=memod_weights, input_mat=str(current),
            threshold=threshold, top_slices=top_slices, var_candidates="ima_comb,ima_comb1",
            disable_onednn=disable_onednn)
        if detection.get("pred_label") not in (0, 1):
            raise ValueError("ME-MoD did not return a binary pred_label")
        stages.append({
            "iteration": iteration,
            "mat_path": str(current),
            "p_subject": detection.get("p_subject"),
            "pred_label": detection["pred_label"],
            "pred_label_name": detection.get("pred_label_name"),
            "threshold": detection.get("threshold"),
            "detection": detection,
            "correction": None if correction is None else {
                "command": correction.get("command"),
                "config": correction.get("persistent_config"),
                "summary_path": correction.get("summary_path"),
            },
        })
        print(f"[MDCC] {subject}: pass {iteration} p_motion={detection.get('p_subject'):.4f} "
              f"-> {detection.get('pred_label_name')}", flush=True)
        if detection["pred_label"] == 0:
            stop_reason = "gate_motion_free"
            break
        if iteration >= max_iters:
            stop_reason = "cap_still_motion_positive"
            break

    corrections = len(stages) - 1
    if corrections > 0:
        final_mat = output_dir / f"ima_comb_{subject}_mdcc.mat"
        shutil.copy2(current, final_mat)
    else:
        final_mat = input_mat
    if not keep_work and work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)
        for stage in stages:
            if stage["iteration"] > 0:
                stage["mat_path"] = None

    report = {
        "input_mat": str(input_mat),
        "subject": subject,
        "max_iters": max_iters,
        "correction_passes": corrections,
        "stop_reason": stop_reason,
        "final_pred_label": stages[-1]["pred_label"],
        "final_pred_label_name": stages[-1]["pred_label_name"],
        "final_mat": str(final_mat),
        "returned_unchanged": corrections == 0,
        "total_seconds": time.perf_counter() - started,
        "models": {
            "memod_config": str(memod_config), "memod_weights": str(memod_weights),
            "memoc_config": str(memoc_config), "memoc_weights": str(memoc_weights),
        },
        "work_dir": str(work_root) if keep_work else None,
        "stages": stages,
    }
    report_path = output_dir / f"{subject}_mdcc_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[MDCC] {subject}: {corrections} correction pass(es), stop={stop_reason}, "
          f"final={report['final_pred_label_name']}. Output: {final_mat}", flush=True)
    print(f"[MDCC] report: {report_path}", flush=True)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="MDCC: detect -> conditionally correct -> re-detect on one volume.")
    p.add_argument("--input_mat", required=True, help="Complex multi-echo MAT volume (H, W, slices, echoes)")
    p.add_argument("--output_dir", default="results/mdcc")
    p.add_argument("--subject", default=None, help="Subject id for output names (default: filename without ima_comb_)")
    p.add_argument("--max_iters", type=int, default=3, help="Maximum correction passes (1-5, paper: 3)")
    p.add_argument("--threshold", type=float, default=None, help="Override the calibrated ME-MoD threshold")
    p.add_argument("--memod_config", default=str(bridge.DEFAULT_MEMOD_CONFIG))
    p.add_argument("--memod_weights", default=str(bridge.DEFAULT_MEMOD_WEIGHTS))
    p.add_argument("--memoc_config", default=str(bridge.DEFAULT_MEMOC_CONFIG))
    p.add_argument("--memoc_weights", default=str(bridge.DEFAULT_MEMOC_WEIGHTS))
    p.add_argument("--top_slices", type=int, default=12, help="Highest-scoring slices listed in the detection report")
    p.add_argument("--batch_size", type=int, default=1, help="ME-MoC inference batch size (slices)")
    p.add_argument("--no_slice_crop", action="store_true", help="Correct all slices instead of the configured slice range")
    p.add_argument("--disable_onednn", action="store_true", help="Set TF_ENABLE_ONEDNN_OPTS=0 for the model processes")
    p.add_argument("--keep_work", action="store_true", help="Keep staged inputs and every intermediate pass")
    args = p.parse_args()
    run_mdcc(
        input_mat=Path(args.input_mat), output_dir=Path(args.output_dir), subject=args.subject,
        max_iters=args.max_iters, threshold=args.threshold,
        memod_config=args.memod_config, memod_weights=args.memod_weights,
        memoc_config=args.memoc_config, memoc_weights=args.memoc_weights,
        top_slices=args.top_slices, batch_size=args.batch_size, no_slice_crop=args.no_slice_crop,
        disable_onednn=args.disable_onednn, keep_work=args.keep_work,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
