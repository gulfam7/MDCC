"""Checks of the motion simulation on a small synthetic volume (no TensorFlow needed).

    python -m unittest tests.test_motion_simulation
"""
import os
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat, savemat

from motion_simulation.simulate_motion import (
    PROFILES,
    corrupt_volume,
    img2ksp,
    ksp2img,
    seeded,
    simulate,
)


def _volume(seed=0, shape=(32, 192, 8, 3)):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


class TestMotionSimulation(unittest.TestCase):
    def test_fourier_round_trip(self):
        vol = _volume()
        self.assertTrue(np.allclose(ksp2img(img2ksp(vol)), vol, atol=1e-5))

    def test_only_listed_lines_change_and_centre_is_kept(self):
        vol = _volume()
        for name, profile in PROFILES.items():
            with seeded(f"{name}_r1"):
                motion, events = corrupt_volume(vol, profile)
            self.assertEqual(motion.shape, vol.shape)
            self.assertTrue(profile.motion_rng[0] <= len(events) <= profile.motion_rng[1])
            changed = np.any(np.abs(img2ksp(motion) - img2ksp(vol)) > 1e-4, axis=(0, 2, 3))
            listed = np.zeros(vol.shape[1], dtype=bool)
            for event in events:
                listed[event["first_line"]:event["end_line_exclusive"]] = True
            self.assertTrue(changed.any())
            self.assertFalse(np.any(changed & ~listed))          # nothing outside the listed lines
            centre = vol.shape[1] // 2
            self.assertFalse(changed[centre - 10:centre + 10].any())

    def test_same_seed_same_motion(self):
        vol = _volume()
        profile = PROFILES["snr3d_mid"]
        with seeded("snr3d_mid_r1"):
            first, events_first = corrupt_volume(vol, profile)
        with seeded("snr3d_mid_r1"):
            second, events_second = corrupt_volume(vol, profile)
        with seeded("snr3d_mid_r2"):
            other, _ = corrupt_volume(vol, profile)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(events_first, events_second)
        self.assertFalse(np.array_equal(first, other))

    def test_files_labels_and_slice_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth_root, motion_root = os.path.join(tmp, "truth"), os.path.join(tmp, "motion")
            vol = _volume(1)
            os.makedirs(os.path.join(truth_root, "s01"))
            savemat(os.path.join(truth_root, "s01", "ima_comb_s01.mat"), {"ima_comb": vol})
            written = simulate(truth_root, motion_root, ["s01"], [PROFILES["snr3d_light"]],
                               repeat_each=2, slice_ranges=[(2, 5)])
            names = sorted(os.path.basename(p) for p in written)
            self.assertEqual(names, ["ima_comb_s01_nrand1-3_snr3d_light_r1.mat",
                                     "ima_comb_s01_nrand1-3_snr3d_light_r2.mat"])
            mat = loadmat(written[0])
            self.assertEqual(mat["slice_label"].ravel().tolist(), [0, 0, 1, 1, 1, 0, 0, 0])
            self.assertEqual(mat["target_slices"].ravel().tolist(), [2, 3, 4])
            self.assertTrue(np.array_equal(mat["ima_comb"][:, :, 0], vol[:, :, 0]))      # copied slice
            self.assertFalse(np.array_equal(mat["ima_comb"][:, :, 3], vol[:, :, 3]))     # motion slice
            self.assertTrue(os.path.isfile(written[0][:-4] + ".motion.json"))


if __name__ == "__main__":
    unittest.main()
