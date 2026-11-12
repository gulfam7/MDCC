"""Installation check: both models build, load the released weights and run.

    python -m unittest tests.test_models
"""
import json
import os
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from memoc.models import build_memoc_from_config, make_img_loss
from memod.models import build_memod

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_json(*parts):
    with open(os.path.join(ROOT, *parts), "r") as f:
        return json.load(f)


class TestMEMoD(unittest.TestCase):
    def test_released_weights(self):
        cfg = _load_json("weights", "memod", "config.json")
        model = build_memod(
            slice_shape=(64, 192, 20),
            model_cfg=cfg["model"],
            aggregation_cfg=cfg["classification"]["aggregation"],
        )
        self.assertEqual(model.count_params(), 718_139)
        model.load_weights(os.path.join(ROOT, "weights", "memod", "best_weights.h5"))
        bag = np.random.default_rng(0).normal(size=(1, 16, 64, 192, 20)).astype(np.float32)
        p_subject, p_slice, subject_logit, slice_logits = model([bag, np.ones((1, 16), np.float32)], training=False)
        self.assertEqual(tuple(p_subject.shape), (1, 1))
        self.assertEqual(tuple(p_slice.shape), (1, 16))
        self.assertTrue(0.0 <= float(p_subject[0, 0]) <= 1.0)
        threshold = _load_json("weights", "memod", "decision_threshold.json")["threshold"]
        self.assertAlmostEqual(threshold, cfg["classification"]["decision_threshold"], places=6)


class TestMEMoC(unittest.TestCase):
    def test_released_weights(self):
        cfg = _load_json("weights", "memoc", "config.json")
        model = build_memoc_from_config(cfg, input_shape=(64, 192, 10, 2))
        self.assertEqual(model.count_params(), 39_395_392)
        model.load_weights(os.path.join(ROOT, "weights", "memoc", "best-psnr.h5"))
        x = np.random.default_rng(0).normal(size=(1, 64, 192, 10, 2)).astype(np.float32)
        y = model(x, training=False).numpy()
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_loss_is_finite(self):
        rng = np.random.default_rng(1)
        y_true = np.abs(rng.normal(size=(2, 16, 16, 10, 2))).astype(np.float32)
        y_pred = np.abs(rng.normal(size=(2, 16, 16, 10, 2))).astype(np.float32)
        loss = make_img_loss(lambda_mag=0.2, lambda_phys=0.02)(y_true, y_pred).numpy()
        self.assertTrue(np.isfinite(loss) and loss > 0)


if __name__ == "__main__":
    unittest.main()
