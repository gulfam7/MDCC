"""Plot per-pass cohort metrics from ``pipeline.evaluate_iterative_cohort`` output.

    python -m pipeline.plot_iterative_metrics --input-dir results/iterative_moderate

This script only replots ``iteration_metric_summary.csv``. It does not rerun
the detector, corrector, or metric calculations.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("ssim", "(a) SSIM ↑"),
    ("nrmse_pct", "(b) NRMSE (%) ↓"),
    ("psnr_db", "(c) PSNR (dB) ↑"),
    ("gradient_energy", "(d) Gradient energy"),
    ("laplacian_variance", "(e) Laplacian variance"),
    ("gradient_error", "(f) Gradient error ↓"),
)

TRAJECTORIES = (
    ("fixed_pass_stress", "Forced-pass stress test", "#D55E00", "s", "--", 2),
    ("gated_policy", "Detector-gated policy", "#0072B2", "o", "-", 3),
)

REQUIRED_COLUMNS = {
    "trajectory", "iteration", "metric", "n", "mean", "ci95_low", "ci95_high"
}


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = {
                    "trajectory": raw["trajectory"],
                    "iteration": int(raw["iteration"]),
                    "metric": raw["metric"],
                    "n": int(raw["n"]),
                    "mean": float(raw["mean"]),
                    "ci95_low": float(raw["ci95_low"]),
                    "ci95_high": float(raw["ci95_high"]),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid value at {path}:{line_number}: {exc}") from exc
            numeric = (row["mean"], row["ci95_low"], row["ci95_high"])
            if not all(math.isfinite(float(value)) for value in numeric):
                continue
            rows.append(row)
    return rows


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def plot(rows: list[dict[str, object]], output_stem: Path, dpi: int,
         width: float, height: float, formats: list[str]) -> list[Path]:
    configure_style()
    available_metrics = {str(row["metric"]) for row in rows}
    missing_metrics = [metric for metric, _ in METRICS if metric not in available_metrics]
    if missing_metrics:
        raise ValueError(f"Summary CSV has no finite data for: {missing_metrics}")

    fig, axes = plt.subplots(2, 3, figsize=(width, height), sharex=True)
    all_iterations = sorted({int(row["iteration"]) for row in rows})
    common_n = sorted({int(row["n"]) for row in rows if str(row["metric"]) in dict(METRICS)})

    for ax, (metric, title) in zip(axes.flat, METRICS):
        for trajectory, label, color, marker, linestyle, zorder in TRAJECTORIES:
            selected = sorted(
                (row for row in rows
                 if row["metric"] == metric and row["trajectory"] == trajectory),
                key=lambda row: int(row["iteration"]),
            )
            if not selected:
                raise ValueError(f"Missing {metric!r} rows for trajectory {trajectory!r}")
            x = np.asarray([int(row["iteration"]) for row in selected], dtype=int)
            mean = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["ci95_low"]) for row in selected])
            high = np.asarray([float(row["ci95_high"]) for row in selected])
            ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0, zorder=zorder - 1)
            ax.plot(
                x, mean, color=color, marker=marker, linestyle=linestyle,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.25,
                label=label, zorder=zorder,
            )

        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks(all_iterations)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.8)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.5, alpha=0.55)
        ax.tick_params(direction="out", length=3, width=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.06, y=0.12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Display the deployed policy first, while plotting it last so coincident
    # pass-0/pass-1 markers remain visible.
    handles, labels = handles[::-1], labels[::-1]
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=2, frameon=False, handlelength=2.8, columnspacing=2.2,
    )
    sample_note = f"Cohort mean and 95% CI; n = {common_n[0]}" if len(common_n) == 1 else "Cohort mean and 95% CI"
    fig.text(0.99, 0.018, sample_note, ha="right", va="bottom", fontsize=7.5, color="#555555")
    fig.supxlabel("Correction pass (0 = motion-corrupted input)", y=0.035, fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.89, wspace=0.28, hspace=0.32)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in formats:
        target = output_stem.with_suffix("." + extension)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if extension in {"png", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        if extension in {"tif", "tiff"}:
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        if extension == "pdf":
            kwargs["metadata"] = {
                "Title": "Iterative detect-correct-detect evaluation",
                "Subject": "Detector-gated and forced-pass correction trajectories",
            }
        fig.savefig(target, **kwargs)
        outputs.append(target)
    plt.close(fig)
    return outputs


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="Folder containing iteration_metric_summary.csv")
    p.add_argument("--csv-name", default="iteration_metric_summary.csv")
    p.add_argument("--output-stem", default="metrics_by_iteration_publication")
    p.add_argument("--formats", nargs="+", choices=("png", "pdf", "tiff"), default=("png", "pdf", "tiff"))
    p.add_argument("--dpi", type=int, default=600, help="Resolution for raster exports")
    p.add_argument("--width", type=float, default=10.0, help="Figure width in inches")
    p.add_argument("--height", type=float, default=6.2, help="Figure height in inches")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.dpi < 300 or args.width <= 0 or args.height <= 0:
        raise ValueError("Use dpi >= 300 and positive figure dimensions")
    input_dir = args.input_dir.resolve()
    csv_path = input_dir / args.csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    output_stem = Path(args.output_stem)
    if not output_stem.is_absolute():
        output_stem = input_dir / output_stem
    outputs = plot(read_rows(csv_path), output_stem, args.dpi, args.width, args.height, list(args.formats))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
