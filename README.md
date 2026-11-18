# MDCC: Motion Detection and Conditional Correction for multi-echo GRE MRI

This repository contains the code and trained models for:

> G. A. Saju, J. Lewis, S. V. V. N. Kothapalli, C. Eldeniz, D. A. Yablonskiy.
> **A Deep Learning Framework for Retrospective Motion Detection and Conditional Correction (MDCC): Application to Multi-Gradient-Recalled Echo Quantitative MRI.**
> *Journal of Imaging Informatics in Medicine*, 2026.
<!-- Add volume, pages and DOI here and in CITATION.cff once they are assigned. -->

MDCC is a retrospective, image-domain pipeline for reconstructed complex multi-echo gradient-recalled-echo (mGRE) volumes. It has two networks:

- **ME-MoD** (Multi-Echo Motion Detection) classifies a whole volume as motion-corrupted or motion-free. Each slice is encoded from its complex echoes with a *paired complex echo gate*, which rescales echo magnitudes without rotating phase. The top-14 slice logits are then averaged into one volume-level probability, compared against a threshold calibrated on the validation set.
- **ME-MoC** (Multi-Echo Motion Correction) is a complex-valued (2+1)D U-Net with echo-axis multi-head self-attention at the bottleneck. It is trained with a combined complex, magnitude and T2*-consistency loss.

Correction is **conditional**: ME-MoC only runs on volumes that ME-MoD flags, and every corrected volume is checked again.

```
input volume ──► ME-MoD ──► motion-free? ──yes──► return the volume unchanged
                   ▲             │ no
                   │             ▼
                   └─────────  ME-MoC   (at most --max_iters correction passes; paper: 3)
```

## Repository layout

```
MDCC/
├── memod/                  ME-MoD detector
│   ├── models.py           paired complex echo gate, slice encoder, top-h aggregator
│   ├── train.py            training + validation threshold calibration
│   ├── test.py             eval / deploy / deploy_folder
│   ├── cache_builder.py    per-volume HDF5 caches (paired, simulated and acquired data)
│   ├── dataset.py          tf.data bag-of-slices pipeline
│   ├── common.py           config defaults, slice labels, input preparation
│   ├── label_slice_ranges.py  slice-label sidecars for acquired motion volumes
│   ├── preprocess.py, mat_io.py
├── memoc/                  ME-MoC corrector
│   ├── models.py           complex (2+1)D U-Net, echo-axis attention, loss
│   ├── train.py            training with checkpoint selection (best validation PSNR)
│   ├── test.py             motion-only inference and paired evaluation
│   ├── cache_builder.py, dataset.py, preprocess.py, mat_io.py
├── pipeline/               MDCC loop
│   ├── run_mdcc.py         detect -> correct -> re-detect on one volume
│   ├── evaluate_iterative_cohort.py  gated policy and fixed-pass stress test on a cohort
│   ├── plot_iterative_metrics.py     per-pass cohort figure
│   └── bridge.py           subprocess interface to ME-MoD and ME-MoC
├── motion_simulation/      simulated motion for the paired training and test data
│   ├── simulate_motion.py  k-space line replacement with rigid shifts and rotations (light, moderate)
│   ├── mat_io.py, LICENSE_LEARN.txt
├── configs/                training configs (memod.json, memoc.json)
├── weights/
│   ├── memod/              best_weights.h5, config.json, decision_threshold.json
│   └── memoc/              best-psnr.h5, config.json
└── tests/
    ├── test_models.py             checks that both models build and load the released weights
    └── test_motion_simulation.py  checks the motion simulation on a small synthetic volume
```

## Installation

Tested with Python 3.9 and TensorFlow 2.15.1 (Python 3.9–3.11 are supported by this TensorFlow release).

```bash
git lfs install            # the ME-MoC checkpoint (159 MB) is stored with Git LFS
git clone <repository-url> MDCC
cd MDCC
python -m pip install -r requirements.txt
python -m unittest tests.test_models
```

A GPU is recommended for training and speeds up inference. Inference also runs on CPU; TensorFlow 2.15 is CPU-only on native Windows (use WSL2 for GPU support).

Run all commands from the repository root. Relative paths in commands and configs are resolved against the repository root.

## Input data

Each volume is a MATLAB `.mat` file (v7 or v7.3) with one complex array named `ima_comb` (or `ima_comb1`):

- **Contents:** coil-combined complex images (real and imaginary parts), not magnitude-only data.
- **Shape:** `(height, width, slices, echoes)`.
- **Released models:** trained on 3D mGRE data with ten echoes (TE₁ = 4 ms, ΔTE = 4 ms), TR = 50 ms, 1 × 1 × 2 mm³ resolution, matrix 256 × 192 × 96, acquired at 3 T.
- **Width:** volumes 190 columns wide are padded to 192 internally and restored on output.

Other protocols need care:

- **Echoes:** the networks are defined for ten echoes. ME-MoC's T2*-consistency loss assumes uniform echo spacing (set `train.physics.use_te_weighting` and `te_ms` otherwise).
- **Matrix:** the in-plane matrix must be divisible by 16.
- **Retraining:** data with a different number of echoes or different echo times require retraining or fine-tuning.
- **Slices:** the slices processed by each model are set by `setting.slice_rng` (zero-based, end-exclusive). ME-MoD scores slices `[29, 75)`. ME-MoC corrects slices `[20, 82)` and copies the remaining slices from its input unchanged. Adjust these ranges for volumes with a different slice count.

## Run MDCC on one volume

```bash
python -m pipeline.run_mdcc --input_mat path/to/ima_comb_SUBJ.mat --output_dir results/mdcc
```

Outputs:

- **`results/mdcc/ima_comb_SUBJ_mdcc.mat`:** the corrected volume (variable `ima_comb`, same shape and intensity scale as the input). It is written only when at least one correction pass ran. A volume classified as motion-free is returned unchanged.
- **`results/mdcc/SUBJ_mdcc_report.json`:** the motion probability and decision for every pass, the stop reason (`gate_motion_free` or `cap_still_motion_positive`) and the exact commands that were run.

Useful options:

- `--max_iters N`: maximum number of correction passes (default 3).
- `--threshold T`: override the calibrated ME-MoD threshold (0.1546).
- `--keep_work`: keep every intermediate pass.
- `--no_slice_crop`: correct all slices.

## Detection or correction only

```bash
# ME-MoD on one volume, or on every ima_comb_*.mat in a folder
python -m memod.test deploy --config weights/memod/config.json --weights weights/memod/best_weights.h5 \
    --input_mat path/to/ima_comb_SUBJ.mat --output_json results/SUBJ_memod.json
python -m memod.test deploy_folder --config weights/memod/config.json --weights weights/memod/best_weights.h5 \
    --input_root path/to/folder --output_json results/memod_folder.json

# ME-MoC on data/motion/SUBJ/ima_comb_SUBJ.mat, without ground truth
python -m memoc.test --config weights/memoc/config.json --weights weights/memoc/best-psnr.h5 \
    --subjects SUBJ --motion_only --motion_root data/motion --motion_label "" --restore_native_width
```

`memod.test` reads the decision threshold from `decision_threshold.json` next to the weights. `memoc.test` writes `<SUBJ>__<label>_corrected.mat` at the input intensity scale and a `test_summary.json` under `results/memoc/`.

## Training

Expected data layout (paths and file templates are set in the `data` block of each config):

```
data/
  truth/<subj>/ima_comb_<subj>.mat           motion-free volumes
  motion/<subj>/ima_comb_<subj><label>.mat   simulated motion-corrupted versions, e.g. label _nrand2-5_snr3d_mid_r2
  real_motion/<subj>/ima_comb_<subj>.mat     acquired motion-corrupted volumes (ME-MoD)
  real_clean/<subj>/ima_comb_<subj>.mat      acquired motion-free volumes (ME-MoD)
```

### Simulating motion

The motion-corrupted volumes under `data/motion/` are simulated from the motion-free volumes under `data/truth/`:

```bash
python -m motion_simulation.simulate_motion --truth_root data/truth --motion_root data/motion
```

For every subject this writes four volumes, two realizations (`r1`, `r2`) of each severity, plus a `.motion.json` file listing the simulated events:

| Severity | Motion label | Events | Lines replaced per event | Shift (pixels) | Rotation (degrees) | Centre gap |
|---|---|---|---|---|---|---|
| Light | `_nrand1-3_snr3d_light_r1`, `_r2` | 1–3 | 1–3 | ±3 | ±2, ±1, ±1 | 12 |
| Moderate | `_nrand2-5_snr3d_mid_r1`, `_r2` | 2–5 | 2–5 | ±6 | ±4, ±2, ±2 | 14 |

These are the ranges of Table 1 of the paper. The simulator works as follows:

- **Events:** each event shifts and rotates the motion-free volume, with the same rigid transform for all echoes.
- **Line replacement:** a band of consecutive k-space lines along the width (phase-encoding) axis is replaced by the lines of the moved volume.
- **Centre gap:** the first line of a band is never drawn within the centre gap around the k-space centre.

Useful options:

- `--subjects S1 S2 ...`: simulate only these subjects (default: all, in sorted order).
- `--profiles snr3d_light snr3d_mid`: severities to simulate.
- `--repeat_each N`: realizations per severity (default 2).
- `--slice_range START END`: keep the motion only in these slices and copy the others from the motion-free volume.

The random generator is seeded once per severity and realization, and its stream continues across the subjects of a run. The motion drawn for a subject therefore depends on its position in the subject list.

The simulation is adapted from the k-space corruption code of LEARN (Xu et al., *Magn Reson Med* 2022; https://github.com/xuxiaojian/2022-MRM-LEARN), which is distributed under the MIT License (`motion_simulation/LICENSE_LEARN.txt`). If you use the simulation, please cite the LEARN paper as well (see [Citation](#citation)).

### Subject lists

The study's participant identifiers are not distributed, so the subject lists in the configs are empty. Before training or paired evaluation, fill them with your own `<subj>` folder names:

- **`configs/memoc.json`:** `data.train_subjects`, `data.valid_subjects` and `data.test_subjects`.
- **`configs/memod.json`:** `data.train_subjects` and `data.valid_subjects` (simulated pairs), plus `data.real_motion_train_subjects`, `data.real_motion_valid_subjects`, `data.real_clean_train_subjects` and `data.real_clean_valid_subjects` (acquired volumes).

Keep every participant in one list only. Running the released models on new volumes does not use these lists.

### Training the models

```bash
# ME-MoD: writes runs/memod/<run>/best_weights.h5, decision_threshold.json, config_used.json, ...
python -m memod.train --config configs/memod.json

# ME-MoC: writes runs/memoc/<run>/model/best-psnr.h5, latest.h5, training_config.json, ...
python -m memoc.train --config configs/memoc.json
```

The configs reproduce the settings of the released models:

- **ME-MoD:** subject batch size 4, Adam with learning rate 1e-4, up to 150 epochs with early stopping on validation AUC, loss weights 1.0 (volume) and 0.15 (slice).
- **ME-MoC:** batch size 2, Adam with learning rate 1e-4 and gradient clipping 1.0, 120 epochs, loss weights 0.78 (complex), 0.2 (magnitude) and 0.02 (T2* consistency), with the checkpoint chosen by best validation PSNR.

Acquired motion volumes can carry per-slice labels for ME-MoD's slice loss. Write them as JSON sidecars with:

```bash
python -m memod.label_slice_ranges --mat data/real_motion/SUBJ/ima_comb_SUBJ.mat --positive-ranges 30-45 52-55
```

Paired evaluation of ME-MoC on `data.test_subjects`:

```bash
python -m memoc.test --config configs/memoc.json --weights weights/memoc/best-psnr.h5 --motion_label _nrand2-5_snr3d_mid_r2
```

Other useful training options:

- **ME-MoC:** `--resume_run`, `--initial_weights`, `--distribution mirrored` (multi-GPU).
- **Both models:** `--cache_root` and `--rebuild_cache` (preprocessed slices are cached under `cache/`).

## Evaluate the iterative loop on a cohort

`pipeline.evaluate_iterative_cohort` runs two procedures on every volume in a folder:

- **Gated policy:** the deployed MDCC loop.
- **Fixed-pass stress test:** correction repeated regardless of the detector.

It saves per-pass detections, metrics, intermediate volumes and figures. The default models are the released checkpoints.

```bash
# simulated motion with aligned ground truth
python -m pipeline.evaluate_iterative_cohort --input-dir data/test_motion_mid \
    --subject-regex "ima_comb_(?P<subject>.+)_nrand2-5_snr3d_mid_r2" \
    --reference-dir data/truth --reference-template "{subject}/ima_comb_{subject}.mat" \
    --max-iters 3 --stress-iters 3 --cohort-label synthetic-moderate --output-dir results/iterative_moderate

# acquired motion, no ground truth: iteration counts and image-change diagnostics only
python -m pipeline.evaluate_iterative_cohort --input-dir data/real_motion_test \
    --max-iters 3 --stress-iters 3 --cohort-label real-motion --output-dir results/iterative_real

python -m pipeline.plot_iterative_metrics --input-dir results/iterative_moderate
```

Notes:

- **Metrics:** reference-based metrics (SSIM, NRMSE, PSNR) are computed on magnitude images inside a foreground mask, thresholded at 3% of the reference echo-1 maximum. A per-slice, per-echo least-squares intensity alignment (`--metric-normalization robust_scale`) is applied first.
- **Slices and ROI:** by default only the corrected slices are evaluated. `--slice-range` (one-based, inclusive) and `--roi X Y W H` restrict the evaluation.
- **Checks:** `--dry-run` validates file matching without running the models.
- **Output folder:** existing non-empty output directories are never overwritten.

## Released weights

| Model | File | Parameters | Notes |
|---|---|---|---|
| ME-MoD | `weights/memod/best_weights.h5` | 718,139 | best validation AUC; threshold 0.1546 in `decision_threshold.json` |
| ME-MoC | `weights/memoc/best-psnr.h5` | 39,395,392 | best validation PSNR |

These are the checkpoints used for the results reported in the paper. The `config.json` next to each file holds the exact model and preprocessing settings; its subject lists are empty.

## Citation

If you use this code or the trained models, please cite:

```bibtex
@article{saju2026mdcc,
  title   = {A Deep Learning Framework for Retrospective Motion Detection and Conditional Correction ({MDCC}): Application to Multi-Gradient-Recalled Echo Quantitative {MRI}},
  author  = {Saju, Gulfam Ahmed and Lewis, Jeramy and Kothapalli, Satya V. V. N. and Eldeniz, Cihat and Yablonskiy, Dmitriy A.},
  journal = {Journal of Imaging Informatics in Medicine},
  year    = {2026}
}
```

If you use the motion simulation, please also cite LEARN:

```bibtex
@article{xu2022learn,
  title   = {Learning-based motion artifact removal networks for quantitative {R2*} mapping},
  author  = {Xu, Xiaojian and Kothapalli, Satya V. V. N. and Liu, Jiaming and Kahali, Sayan and Gan, Weijie and Yablonskiy, Dmitriy A. and Kamilov, Ulugbek S.},
  journal = {Magnetic Resonance in Medicine},
  volume  = {88},
  number  = {1},
  pages   = {106--119},
  year    = {2022},
  doi     = {10.1002/mrm.29188}
}
```

## Acknowledgments

This work was supported by NIH/NIA grants RF1AG077658, RF1AG082030, R01AG077658 and R01AG082030.

## Contact

Gulfam Ahmed Saju (gsaju@wustl.edu) and Dmitriy A. Yablonskiy (yablonskiyd@wustl.edu), Mallinckrodt Institute of Radiology, Washington University in St. Louis.
