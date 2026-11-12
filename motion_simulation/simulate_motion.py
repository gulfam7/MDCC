"""Simulate rigid head motion in motion-free complex multi-echo GRE volumes.

    python -m motion_simulation.simulate_motion --truth_root data/truth --motion_root data/motion

This is the simulation used to create the paired training and test data of
the MDCC paper. For every motion-free volume

    <truth_root>/<subj>/ima_comb_<subj>.mat

it writes, for each profile and repetition,

    <motion_root>/<subj>/ima_comb_<subj>_nrand<a>-<b>_<profile>_r<n>.mat
    <motion_root>/<subj>/ima_comb_<subj>_nrand<a>-<b>_<profile>_r<n>.motion.json

for example ``ima_comb_<subj>_nrand2-5_snr3d_mid_r1.mat``. The part after
``ima_comb_<subj>`` is the motion label used in the configs.

Method (k-space line replacement, LEARN "snr3d" simulation)
-----------------------------------------------------------
1. The volume ``(height, width, slices, echoes)`` is transformed to 2D
   k-space along height and width.
2. A random number of motion events is drawn. For each event the original
   volume is shifted in-plane and rotated about three axes (the same rigid
   transform for all echoes), and transformed to k-space.
3. A few consecutive k-space lines along the width (phase-encoding) axis
   are replaced, for all slices and echoes, by the lines of the moved
   volume. Start positions near the k-space centre are excluded.
4. The corrupted k-space is transformed back to image space.

Profiles (Table 1 of the paper)
-------------------------------
``snr3d_light`` (light):   1-3 events, 1-3 lines each, shifts within +/-3
    pixels, in-plane rotation within +/-2 degrees, through-plane rotations
    within +/-1 degree, start positions exclude the central +/-12 lines.
``snr3d_mid`` (moderate):  2-5 events, 2-5 lines each, shifts within +/-6
    pixels, in-plane rotation within +/-4 degrees, through-plane rotations
    within +/-2 degrees, start positions exclude the central +/-14 lines.

Reproducibility
---------------
The random generator is seeded once per profile and repetition, from the
text ``<profile>_r<n>``, and its stream continues across the subjects of
one run. The motion drawn for a subject therefore depends on the profile,
the repetition, and the position of the subject in the subject list
(sorted folder names unless ``--subjects`` is given).

Attribution
-----------
``img2ksp``, ``ksp2img``, ``shift_and_rotate`` and the line-replacement
loop in ``corrupt_volume`` are taken from the LEARN code of Xu et al.
(https://github.com/xuxiaojian/2022-MRM-LEARN, MIT License, Copyright (c)
2022 Xiaojian Xu; see LICENSE_LEARN.txt in this folder). If you use this
simulation, please also cite:

    X. Xu, S. V. V. N. Kothapalli, J. Liu, S. Kahali, W. Gan, D. A. Yablonskiy,
    U. S. Kamilov. Learning-based motion artifact removal networks for
    quantitative R2* mapping. Magn Reson Med 2022;88(1):106-119.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import numpy.fft as nf
import scipy.ndimage
from scipy.io import savemat

from .mat_io import load_mat_var

PHASE_ENCODING_LINES = 192


@dataclass(frozen=True)
class MotionProfile:
    """Parameter ranges of one motion severity (all ranges are inclusive)."""

    name: str
    motion_rng: Tuple[int, int]      # number of motion events
    width_rng: Tuple[int, int]       # consecutive k-space lines replaced per event
    shift_h_rng: Tuple[int, int]     # shift along height, pixels
    shift_w_rng: Tuple[int, int]     # shift along width, pixels
    rotate_rng_hw: Tuple[int, int]   # rotation in the height-width plane, degrees
    rotate_rng_hs: Tuple[int, int]   # rotation in the height-slice plane, degrees
    rotate_rng_ws: Tuple[int, int]   # rotation in the width-slice plane, degrees
    center_gap: int                  # start positions exclude the central +/- center_gap lines

    @property
    def label(self) -> str:
        return f"nrand{self.motion_rng[0]}-{self.motion_rng[1]}_{self.name}"


PROFILES: Dict[str, MotionProfile] = {
    "snr3d_light": MotionProfile(
        name="snr3d_light",
        motion_rng=(1, 3),
        width_rng=(1, 3),
        shift_h_rng=(-3, 3),
        shift_w_rng=(-3, 3),
        rotate_rng_hw=(-2, 2),
        rotate_rng_hs=(-1, 1),
        rotate_rng_ws=(-1, 1),
        center_gap=12,
    ),
    "snr3d_mid": MotionProfile(
        name="snr3d_mid",
        motion_rng=(2, 5),
        width_rng=(2, 5),
        shift_h_rng=(-6, 6),
        shift_w_rng=(-6, 6),
        rotate_rng_hw=(-4, 4),
        rotate_rng_hs=(-2, 2),
        rotate_rng_ws=(-2, 2),
        center_gap=14,
    ),
}

DEFAULT_PROFILES = ["snr3d_light", "snr3d_mid"]


# --------------------------------------------------------------------------- LEARN functions
def img2ksp(img: np.ndarray) -> np.ndarray:
    """Image -> 2D k-space along height (axis 0) and width (axis 1)."""
    nheight = img.shape[0]
    nwidth = img.shape[1]
    imatmp2 = nf.fftshift(nf.ifft(nf.fftshift(img, axes=1), n=nwidth, axis=1), axes=1)
    ksp = nf.fftshift(nf.fft(nf.fftshift(imatmp2, axes=0), n=nheight, axis=0), axes=0)
    return ksp


def ksp2img(ksp: np.ndarray) -> np.ndarray:
    """Inverse of :func:`img2ksp`."""
    nheight = ksp.shape[0]
    nwidth = ksp.shape[1]
    imatmp2 = nf.fftshift(nf.fft(nf.fftshift(ksp, axes=1), n=nwidth, axis=1), axes=1)
    ipt = nf.fftshift(nf.ifft(nf.fftshift(imatmp2, axes=0), n=nheight, axis=0), axes=0)
    return ipt


def shift_and_rotate(img: np.ndarray, shift: Sequence[int], rotate: Sequence[int], mode: str = "constant") -> np.ndarray:
    """Shift and rotate a complex ``(height, width, slices, echoes)`` volume.

    ``shift`` is given per axis in pixels; ``rotate`` holds the angles in
    degrees in the height-width, height-slice and width-slice planes. Real
    and imaginary parts are interpolated separately (cubic spline, zeros
    outside the volume).
    """
    tmp_img_real = img.real
    tmp_img_imag = img.imag
    if not all(v == 0 for v in shift):
        tmp_img_real = scipy.ndimage.shift(tmp_img_real, shift, mode=mode, cval=0.0)
        tmp_img_imag = scipy.ndimage.shift(tmp_img_imag, shift, mode=mode, cval=0.0)
    if rotate[0] != 0: tmp_img_real = scipy.ndimage.rotate(tmp_img_real, rotate[0], mode=mode, cval=0.0, axes=(0, 1), reshape=False)
    if rotate[1] != 0: tmp_img_real = scipy.ndimage.rotate(tmp_img_real, rotate[1], mode=mode, cval=0.0, axes=(0, 2), reshape=False)
    if rotate[2] != 0: tmp_img_real = scipy.ndimage.rotate(tmp_img_real, rotate[2], mode=mode, cval=0.0, axes=(1, 2), reshape=False)

    if rotate[0] != 0: tmp_img_imag = scipy.ndimage.rotate(tmp_img_imag, rotate[0], mode=mode, cval=0.0, axes=(0, 1), reshape=False)
    if rotate[1] != 0: tmp_img_imag = scipy.ndimage.rotate(tmp_img_imag, rotate[1], mode=mode, cval=0.0, axes=(0, 2), reshape=False)
    if rotate[2] != 0: tmp_img_imag = scipy.ndimage.rotate(tmp_img_imag, rotate[2], mode=mode, cval=0.0, axes=(1, 2), reshape=False)
    return tmp_img_real + 1j * tmp_img_imag


def _rand_int(data_range: Sequence[int]) -> int:
    """Random integer from the inclusive range, drawn from NumPy's global generator.

    Equivalent to the ``np.random.random_integers(low, high)`` call of the
    LEARN code, so the same seed gives the same motion.
    """
    return int(np.random.randint(int(data_range[0]), int(data_range[1]) + 1, dtype="l"))


# --------------------------------------------------------------------------- simulation
def start_candidates(profile: MotionProfile, pe_lines: int = PHASE_ENCODING_LINES) -> np.ndarray:
    """Allowed first lines of a replaced block (outside the k-space centre)."""
    return np.concatenate(
        (
            np.arange(0, pe_lines // 2 - profile.center_gap),
            np.arange(pe_lines // 2 + profile.center_gap, pe_lines - 1),
        ),
        axis=0,
    )


def corrupt_volume(
    truth: np.ndarray, profile: MotionProfile, pe_lines: int = PHASE_ENCODING_LINES
) -> Tuple[np.ndarray, List[Dict]]:
    """Return the motion-corrupted volume and the list of simulated events.

    ``truth`` is a complex ``(height, width, slices, echoes)`` array. Random
    numbers come from NumPy's global generator (see :func:`seeded`).
    """
    if truth.ndim != 4:
        raise ValueError(f"Expected a (height, width, slices, echoes) array, got shape {truth.shape}.")
    candidates = start_candidates(profile, pe_lines)
    ksp_corrupt = img2ksp(truth).copy()

    events: List[Dict] = []
    nmotion = _rand_int(profile.motion_rng)
    for _ in range(nmotion):
        shift = [_rand_int(profile.shift_h_rng), _rand_int(profile.shift_w_rng), 0, 0]
        rotate = [
            _rand_int(profile.rotate_rng_hw),
            _rand_int(profile.rotate_rng_hs),
            _rand_int(profile.rotate_rng_ws),
        ]
        ksp_moved = img2ksp(shift_and_rotate(truth, shift, rotate))
        w_start = int(candidates[_rand_int([0, len(candidates) - 1])])
        w_end = w_start + _rand_int(profile.width_rng)
        ksp_corrupt[:, w_start:w_end, :, :] = ksp_moved[:, w_start:w_end, :, :]
        events.append(
            {
                "shift_height_px": shift[0],
                "shift_width_px": shift[1],
                "rotate_height_width_deg": rotate[0],
                "rotate_height_slice_deg": rotate[1],
                "rotate_width_slice_deg": rotate[2],
                "first_line": w_start,
                "end_line_exclusive": w_end,
            }
        )
    return ksp2img(ksp_corrupt), events


def seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


@contextlib.contextmanager
def seeded(text: str) -> Iterator[None]:
    """Seed NumPy's global generator from ``text`` and restore its state afterwards."""
    original_state = np.random.get_state()
    np.random.seed(seed_from_text(text))
    try:
        yield
    finally:
        np.random.set_state(original_state)


def slice_keep_mask(nslice: int, slice_ranges: Optional[Sequence[Sequence[int]]]) -> np.ndarray:
    """Boolean mask of the slices that keep the simulated motion (all if no range is given)."""
    if not slice_ranges:
        return np.ones((nslice,), dtype=bool)
    keep = np.zeros((nslice,), dtype=bool)
    for pair in slice_ranges:
        if len(pair) != 2:
            raise ValueError(f"slice_range must contain exactly two integers, got {pair}")
        start, end = max(0, int(pair[0])), min(nslice, int(pair[1]))
        if end > start:
            keep[start:end] = True
    if not np.any(keep):
        raise ValueError(f"Slice ranges {slice_ranges} select no slice of a volume with {nslice} slices.")
    return keep


def discover_subjects(truth_root: str) -> List[str]:
    subjects: List[str] = []
    for name in sorted(os.listdir(truth_root)):
        if os.path.isfile(os.path.join(truth_root, name, f"ima_comb_{name}.mat")):
            subjects.append(name)
    return subjects


def simulate(
    truth_root: str,
    motion_root: str,
    subjects: Sequence[str],
    profiles: Sequence[MotionProfile],
    repeat_each: int = 2,
    slice_ranges: Optional[Sequence[Sequence[int]]] = None,
    pe_lines: int = PHASE_ENCODING_LINES,
) -> List[str]:
    """Write one motion-corrupted volume per subject, profile and repetition."""
    written: List[str] = []
    for profile in profiles:
        for repeat in range(1, int(repeat_each) + 1):
            label = f"{profile.label}_r{repeat}"
            with seeded(f"{profile.name}_r{repeat}"):
                for subject in subjects:
                    truth_path = os.path.join(truth_root, subject, f"ima_comb_{subject}.mat")
                    truth = np.asarray(load_mat_var(truth_path, ("ima_comb1", "ima_comb")))
                    if truth.shape[1] != pe_lines:
                        print(
                            f"[warning] {subject}: width is {truth.shape[1]} but --pe_lines is {pe_lines}; "
                            "replaced lines are drawn for the --pe_lines grid."
                        )
                    motion, events = corrupt_volume(truth, profile, pe_lines)

                    nslice = int(truth.shape[2])
                    keep = slice_keep_mask(nslice, slice_ranges)
                    if slice_ranges:
                        motion = np.where(keep.reshape(1, 1, nslice, 1), motion, truth)

                    out_dir = os.path.join(motion_root, subject)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"ima_comb_{subject}_{label}.mat")
                    savemat(
                        out_path,
                        {
                            "ima_comb": motion,
                            "slice_label": keep.astype(np.float32),
                            "target_slices": np.where(keep)[0].astype(np.int32),
                            "motion_backend": np.array("3d"),
                            "motion_profile": np.array(profile.name),
                            "motion_repeat": np.array(int(repeat), dtype=np.int32),
                        },
                    )
                    with open(out_path[: -len(".mat")] + ".motion.json", "w") as f:
                        json.dump(
                            {
                                "subject": subject,
                                "profile": profile.name,
                                "repeat": repeat,
                                "seed_text": f"{profile.name}_r{repeat}",
                                "phase_encoding_lines": pe_lines,
                                "slice_ranges": [list(map(int, r)) for r in slice_ranges] if slice_ranges else None,
                                "events": events,
                            },
                            f,
                            indent=2,
                        )
                    print(f"[motion] {out_path}  ({len(events)} events)")
                    written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate motion-corrupted volumes from motion-free volumes.")
    parser.add_argument("--truth_root", default=os.path.join("data", "truth"), help="folder with <subj>/ima_comb_<subj>.mat")
    parser.add_argument("--motion_root", default=os.path.join("data", "motion"), help="output folder")
    parser.add_argument("--subjects", nargs="*", default=None, help="subject folder names (default: all, sorted)")
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES, choices=sorted(PROFILES))
    parser.add_argument("--repeat_each", type=int, default=2, help="independent realizations per profile (r1, r2, ...)")
    parser.add_argument(
        "--slice_range",
        action="append",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="keep the simulated motion only in slices [START, END) (zero-based); other slices are copied "
        "from the motion-free volume. Repeat the option for several ranges.",
    )
    parser.add_argument(
        "--pe_lines",
        type=int,
        default=PHASE_ENCODING_LINES,
        help="number of k-space lines along the width axis used to draw the replaced lines (paper: 192)",
    )
    args = parser.parse_args()

    truth_root = os.path.abspath(args.truth_root)
    motion_root = os.path.abspath(args.motion_root)
    if not os.path.isdir(truth_root):
        raise FileNotFoundError(f"truth root not found: {truth_root}")
    subjects = list(args.subjects) if args.subjects else discover_subjects(truth_root)
    if not subjects:
        raise ValueError(f"No <subj>/ima_comb_<subj>.mat found in {truth_root}")

    profiles = [PROFILES[name] for name in args.profiles]
    print(
        f"[motion] subjects={len(subjects)} profiles={[p.name for p in profiles]} "
        f"repeat_each={args.repeat_each} slice_ranges={args.slice_range}"
    )
    simulate(truth_root, motion_root, subjects, profiles, args.repeat_each, args.slice_range, args.pe_lines)
    print("[motion] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
