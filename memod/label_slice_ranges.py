"""Write per-slice motion labels for acquired volumes as a JSON sidecar.

ME-MoD's slice-level loss uses these labels for acquired (real) motion
volumes. The sidecar ``<mat-stem>.slice_labels.json`` is placed next to the
MAT file and is read automatically by the cache builder.

Example (1-based slice numbers; slices outside the ranges are 'unknown'):

    python -m memod.label_slice_ranges --mat data/real_motion/021/ima_comb_021.mat \
        --positive-ranges 30-45 52-55 --negative-ranges 60-70
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .mat_io import load_mat_var


def _parse_ranges(items: Sequence[str], index_base: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for raw_item in items:
        for chunk in str(raw_item).split(","):
            token = chunk.strip()
            if not token:
                continue
            if "-" in token:
                start_s, end_s = token.split("-", 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
            else:
                start = int(token)
                end = int(token)
            if end < start:
                raise ValueError(f"Invalid range '{token}': end < start")
            start0 = start - index_base
            end0 = end - index_base
            if start0 < 0 or end0 < 0:
                raise ValueError(
                    f"Range '{token}' becomes negative with index_base={index_base}. "
                    "Use --index-base 0 for 0-based slices or --index-base 1 for human numbering."
                )
            ranges.append((start0, end0))
    return ranges


def _infer_slice_count(mat_path: str, preferred_keys: Iterable[str]) -> int:
    arr = np.asarray(load_mat_var(mat_path, preferred_keys=preferred_keys))
    if arr.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions in {mat_path}, got shape {arr.shape}")
    return int(arr.shape[2])


def _apply_ranges(labels: np.ndarray, ranges: Sequence[Tuple[int, int]], value: float) -> None:
    for start, end in ranges:
        if end >= labels.shape[0]:
            raise ValueError(
                f"Range {start}-{end} exceeds slice count {labels.shape[0]}. "
                "Double-check the numbering and whether you meant 0-based or 1-based slices."
            )
        labels[start : end + 1] = float(value)


def _default_fill(outside_label: str) -> float:
    mode = str(outside_label).strip().lower()
    if mode in ("unknown", "-1"):
        return -1.0
    if mode in ("clean", "0"):
        return 0.0
    raise ValueError(f"Unsupported outside label '{outside_label}'. Use 'unknown'/'-1' or 'clean'/'0'.")


def _label_one_file(
    mat_path: Path,
    *,
    preferred_keys: Sequence[str],
    positive_ranges: Sequence[Tuple[int, int]],
    negative_ranges: Sequence[Tuple[int, int]],
    outside_label: str,
    index_base: int,
    output_path: Path,
) -> None:
    n_slices = _infer_slice_count(str(mat_path), preferred_keys=preferred_keys)
    labels = np.full((n_slices,), _default_fill(outside_label), dtype=np.float32)
    _apply_ranges(labels, negative_ranges, 0.0)
    _apply_ranges(labels, positive_ranges, 1.0)

    target_slices = np.where(labels > 0.5)[0].astype(np.int32)
    payload = {
        "version": 1,
        "mat_file": mat_path.name,
        "slice_count": int(n_slices),
        "index_base_used_for_input_ranges": int(index_base),
        "outside_label": str(outside_label),
        "positive_ranges_0_based_inclusive": [[int(a), int(b)] for a, b in positive_ranges],
        "negative_ranges_0_based_inclusive": [[int(a), int(b)] for a, b in negative_ranges],
        "slice_label": labels.astype(float).tolist(),
        "target_slices": target_slices.astype(int).tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    known = int(np.sum(labels != -1.0))
    positive = int(np.sum(labels > 0.5))
    negative = int(np.sum(labels == 0.0))
    unknown = int(np.sum(labels == -1.0))
    print(f"Wrote {output_path}")
    print(
        "Summary: "
        f"slices={n_slices}, motion={positive}, clean={negative}, "
        f"unknown={unknown}, known={known}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a slice-label JSON sidecar for acquired MAT volumes.")
    parser.add_argument("--mat", required=True, help="One MAT volume or a folder containing MAT volumes")
    parser.add_argument(
        "--positive-ranges",
        nargs="+",
        required=True,
        help="Inclusive motion slice ranges, e.g. 30-45 52-55 or '30-45,52-55'",
    )
    parser.add_argument(
        "--negative-ranges",
        nargs="*",
        default=[],
        help="Optional inclusive motion-free slice ranges to set explicitly to 0.",
    )
    parser.add_argument(
        "--index-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Slice numbering used in the provided ranges. Default: 1.",
    )
    parser.add_argument(
        "--outside-label",
        choices=["unknown", "clean", "-1", "0"],
        default="unknown",
        help="Label for slices not covered by any range: unknown/-1 or clean/0. Default: unknown.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path in single-file mode. Default: <mat-stem>.slice_labels.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sidecar destination in folder mode (input layout is preserved). Default: beside each MAT file.",
    )
    parser.add_argument("--pattern", default="*.mat", help="MAT filename pattern in folder mode (default: *.mat)")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders when --mat is a directory")
    parser.add_argument(
        "--preferred-key",
        action="append",
        dest="preferred_keys",
        default=None,
        help="Preferred MAT variable used to infer the slice count. Repeatable.",
    )
    args = parser.parse_args()

    input_path = Path(args.mat).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"MAT path not found: {input_path}")
    if input_path.is_dir() and args.output:
        raise ValueError("--output is only valid when --mat points to one MAT file")
    if input_path.is_file() and args.output_dir:
        raise ValueError("--output-dir is only valid when --mat points to a directory")

    if input_path.is_dir():
        iterator = input_path.rglob(args.pattern) if args.recursive else input_path.glob(args.pattern)
        mat_paths = sorted(
            (path.resolve() for path in iterator if path.is_file()),
            key=lambda path: path.as_posix().lower(),
        )
    else:
        if input_path.suffix.lower() != ".mat":
            raise ValueError(f"Expected a .mat file, received: {input_path}")
        mat_paths = [input_path]
    if not mat_paths:
        raise FileNotFoundError(f"No MAT files matching {args.pattern!r} found in {input_path}")

    positive_ranges = _parse_ranges(args.positive_ranges, index_base=int(args.index_base))
    negative_ranges = _parse_ranges(args.negative_ranges, index_base=int(args.index_base))
    preferred_keys = tuple(args.preferred_keys or ["ima_comb", "ima_comb1"])
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    for mat_path in mat_paths:
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
        elif output_root is not None:
            relative = mat_path.relative_to(input_path)
            output_path = output_root / relative.parent / f"{mat_path.stem}.slice_labels.json"
        else:
            output_path = mat_path.with_name(f"{mat_path.stem}.slice_labels.json")
        _label_one_file(
            mat_path,
            preferred_keys=preferred_keys,
            positive_ranges=positive_ranges,
            negative_ranges=negative_ranges,
            outside_label=args.outside_label,
            index_base=int(args.index_base),
            output_path=output_path,
        )

    print(f"Completed {len(mat_paths)} MAT file(s).")


if __name__ == "__main__":
    main()
