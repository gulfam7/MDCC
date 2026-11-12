"""Train ME-MoD, the volume-level multi-echo motion detector.

Example (from the repository root):

    python -m memod.train --config configs/memod.json

The run directory (default ``runs/memod/<dataset>_<timestamp>``) receives
``best_weights.h5`` (highest validation AUC), ``latest_weights.h5``,
``final_weights.h5``, ``final.keras``, ``config_used.json``, ``train_log.csv``,
``history.json`` and ``decision_threshold.json``. The decision threshold is
selected on the validation set by balanced accuracy, with specificity as the
tie-break, and is also written to ``classification.decision_threshold`` in
``config_used.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import tensorflow as tf

from .cache_builder import build_cache_for_mode
from .common import abs_from_repo, load_config, save_json
from .dataset import (
    compute_subject_class_weights,
    count_subject_samples,
    infer_sample_shape,
    make_tf_subject_dataset,
)
from .models import build_memod


def _setup_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


class _BaseModelWeightsCheckpoint(tf.keras.callbacks.Callback):
    """Saves the weights of the wrapped ME-MoD model (not the training wrapper)."""

    def __init__(
        self,
        filepath: str,
        target_model: tf.keras.Model,
        monitor: str,
        mode: str,
        save_best_only: bool,
        verbose: int = 0,
    ):
        super().__init__()
        self.filepath = filepath
        self.target_model = target_model
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.save_best_only = bool(save_best_only)
        self.verbose = int(verbose)
        if self.mode == "max":
            self.monitor_op = lambda a, b: a > b
            self.best = -np.inf
        elif self.mode == "min":
            self.monitor_op = lambda a, b: a < b
            self.best = np.inf
        else:
            raise ValueError(f"Unsupported monitor mode: {mode}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        should_save = not self.save_best_only
        current = logs.get(self.monitor)
        if self.save_best_only:
            if current is None:
                return
            if self.monitor_op(float(current), float(self.best)):
                self.best = float(current)
                should_save = True
        if not should_save:
            return
        self.target_model.save_weights(self.filepath)
        if self.verbose:
            status = "improved" if self.save_best_only else "saved"
            print(f"\nEpoch {epoch + 1}: {self.monitor} {status}, writing ME-MoD weights to {self.filepath}")


class RuntimeTracker(tf.keras.callbacks.Callback):
    """Records per-epoch and cumulative wall time in ``logs`` (written to train_log.csv)."""

    def __init__(self, batch_size: int, steps_per_epoch: int):
        super().__init__()
        self.batch_size = max(1, int(batch_size))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self._epoch_start = 0.0
        self._train_start = 0.0

    def on_train_begin(self, logs=None):
        self._train_start = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = float(time.perf_counter() - self._epoch_start)
        total_time = float(time.perf_counter() - self._train_start)
        step_ms = 1000.0 * epoch_time / float(self.steps_per_epoch)
        samples_per_sec = float(self.steps_per_epoch * self.batch_size) / max(epoch_time, 1e-6)
        # CSVLogger runs after this callback and writes these columns.
        logs["epoch_time_sec"] = epoch_time
        logs["total_time_sec"] = total_time
        logs["step_time_ms"] = step_ms
        logs["samples_per_sec"] = samples_per_sec
        print(
            f"[ME-MoD] epoch_runtime: epoch={epoch + 1} sec={epoch_time:.2f} "
            f"total_sec={total_time:.2f} step_ms={step_ms:.2f} samples_per_sec={samples_per_sec:.2f}"
        )

    def on_train_end(self, logs=None):
        total_time = float(time.perf_counter() - self._train_start)
        print(f"[ME-MoD] training complete: total_runtime_sec={total_time:.2f} ({total_time / 3600.0:.3f} h)")


def _build_callbacks(
    out_dir: str,
    target_model: tf.keras.Model,
    monitor: str,
    mode: str,
    patience: int,
    batch_size: int,
    steps_per_epoch: int,
) -> List[tf.keras.callbacks.Callback]:
    return [
        RuntimeTracker(batch_size=batch_size, steps_per_epoch=steps_per_epoch),
        _BaseModelWeightsCheckpoint(
            filepath=os.path.join(out_dir, "best_weights.h5"),
            target_model=target_model,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        _BaseModelWeightsCheckpoint(
            filepath=os.path.join(out_dir, "latest_weights.h5"),
            target_model=target_model,
            monitor=monitor,
            mode=mode,
            save_best_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=int(patience),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=max(4, int(patience) // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "train_log.csv")),
    ]


def _resolve_monitor(train_cfg: Dict) -> Tuple[str, str]:
    requested = str(train_cfg.get("monitor", "") or "").strip() or "val_auc"
    if requested == "val_subject_auc":
        requested = "val_auc"
    return requested, str(train_cfg.get("monitor_mode", "max"))


class SubjectGateTrainer(tf.keras.Model):
    """Training wrapper with the dual-task objective.

    total = w_subject * weighted_BCE(volume label, volume logit)
          + w_slice   * masked_BCE(slice labels, slice logits)

    Slices that are padding, unavailable, or marked unknown do not contribute
    to the slice term.
    """

    def __init__(self, base_model: tf.keras.Model, cfg: Dict, class_weight: Optional[Dict[int, float]] = None):
        super().__init__(name="SubjectGateTrainer")
        self.base_model = base_model
        self.class_weight = class_weight or {}
        self.subject_loss_weight = float(cfg.get("subject_loss_weight", 1.0))
        self.slice_loss_weight = float(cfg.get("slice_loss_weight", 0.15))
        self.unknown_slice_value = float(cfg.get("slice_unknown_value", -1.0))

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.subject_loss_tracker = tf.keras.metrics.Mean(name="subject_loss")
        self.slice_loss_tracker = tf.keras.metrics.Mean(name="slice_loss")
        self.auc_metric = tf.keras.metrics.AUC(name="auc", curve="ROC")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.subject_loss_tracker,
            self.slice_loss_tracker,
            self.auc_metric,
        ]

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def _weighted_subject_bce(self, y_true: tf.Tensor, subject_logit: tf.Tensor) -> tf.Tensor:
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))
        subject_logit = tf.reshape(tf.cast(subject_logit, tf.float32), (-1, 1))
        per = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=subject_logit)
        if self.class_weight:
            w0 = tf.constant(float(self.class_weight.get(0, 1.0)), dtype=per.dtype)
            w1 = tf.constant(float(self.class_weight.get(1, 1.0)), dtype=per.dtype)
            weights = y_true * w1 + (1.0 - y_true) * w0
            per = per * weights
        return tf.reduce_mean(per)

    def _masked_slice_bce(
        self,
        y_true_slice: tf.Tensor,
        slice_logits: tf.Tensor,
        mask: tf.Tensor,
        y_slice_available: tf.Tensor,
    ) -> tf.Tensor:
        y_true_slice = tf.cast(y_true_slice, tf.float32)
        slice_logits = tf.cast(slice_logits, tf.float32)
        mask = tf.cast(mask, tf.float32)
        y_slice_available = tf.cast(y_slice_available, tf.float32)
        unknown_mask = tf.cast(tf.not_equal(y_true_slice, self.unknown_slice_value), tf.float32)
        valid = mask * y_slice_available * unknown_mask
        safe_y = tf.where(valid > 0.5, y_true_slice, tf.zeros_like(y_true_slice))
        per = tf.nn.sigmoid_cross_entropy_with_logits(labels=safe_y, logits=slice_logits)
        per = per * valid
        denom = tf.reduce_sum(valid) + 1e-8
        return tf.reduce_sum(per) / denom

    def train_step(self, data):
        (x, mask, y_slice, y_slice_available), y_subject = data
        with tf.GradientTape() as tape:
            p_subject, p_slice, subject_logit, slice_logits = self.base_model([x, mask], training=True)
            subject_loss = self._weighted_subject_bce(y_subject, subject_logit)
            slice_loss = self._masked_slice_bce(y_slice, slice_logits, mask, y_slice_available)
            reg_loss = tf.add_n(self.base_model.losses) if self.base_model.losses else 0.0
            total_loss = self.subject_loss_weight * subject_loss + self.slice_loss_weight * slice_loss + reg_loss
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.subject_loss_tracker.update_state(subject_loss)
        self.slice_loss_tracker.update_state(slice_loss)
        y_true = tf.reshape(tf.cast(y_subject, tf.float32), (-1, 1))
        self.auc_metric.update_state(y_true, p_subject)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (x, mask, y_slice, y_slice_available), y_subject = data
        p_subject, p_slice, subject_logit, slice_logits = self.base_model([x, mask], training=False)
        subject_loss = self._weighted_subject_bce(y_subject, subject_logit)
        slice_loss = self._masked_slice_bce(y_slice, slice_logits, mask, y_slice_available)
        reg_loss = tf.add_n(self.base_model.losses) if self.base_model.losses else 0.0
        total_loss = self.subject_loss_weight * subject_loss + self.slice_loss_weight * slice_loss + reg_loss

        self.total_loss_tracker.update_state(total_loss)
        self.subject_loss_tracker.update_state(subject_loss)
        self.slice_loss_tracker.update_state(slice_loss)
        y_true = tf.reshape(tf.cast(y_subject, tf.float32), (-1, 1))
        self.auc_metric.update_state(y_true, p_subject)
        return {m.name: m.result() for m in self.metrics}


def _match_subject_shape(x_subject: np.ndarray, slice_shape: Tuple[int, int, int]) -> np.ndarray:
    x_subject = np.asarray(x_subject, dtype=np.float32)
    target_h, target_w, target_c = [int(v) for v in slice_shape]
    cur_h = int(x_subject.shape[1])
    cur_w = int(x_subject.shape[2])
    off_h = max(0, (cur_h - target_h) // 2)
    off_w = max(0, (cur_w - target_w) // 2)
    end_h = off_h + min(cur_h, target_h)
    end_w = off_w + min(cur_w, target_w)
    x_subject = x_subject[:, off_h:end_h, off_w:end_w, :]
    if x_subject.shape[1] < target_h or x_subject.shape[2] < target_w or x_subject.shape[3] < target_c:
        pad_h = max(0, target_h - int(x_subject.shape[1]))
        pad_w = max(0, target_w - int(x_subject.shape[2]))
        pad_c = max(0, target_c - int(x_subject.shape[3]))
        x_subject = np.pad(
            x_subject,
            ((0, 0), (0, pad_h), (0, pad_w), (0, pad_c)),
            mode="constant",
            constant_values=0.0,
        )
    return x_subject[:, :target_h, :target_w, :target_c]


def _predict_subject_probability(model: tf.keras.Model, x_subject: np.ndarray, slice_shape: Tuple[int, int, int]) -> float:
    x_subject = _match_subject_shape(x_subject, slice_shape)
    mask = np.ones((1, int(x_subject.shape[0])), dtype=np.float32)
    out = model([x_subject[np.newaxis, ...], mask], training=False)
    return float(np.asarray(out[0]).reshape(-1)[0])


def _confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, int]:
    yt = (np.asarray(y_true).reshape(-1) > 0.5).astype(np.int32)
    yp = (np.asarray(y_prob).reshape(-1) >= float(thr)).astype(np.int32)
    return {
        "tp": int(np.sum((yp == 1) & (yt == 1))),
        "tn": int(np.sum((yp == 0) & (yt == 0))),
        "fp": int(np.sum((yp == 1) & (yt == 0))),
        "fn": int(np.sum((yp == 0) & (yt == 1))),
    }


def _balanced_accuracy(conf: Dict[str, int]) -> float:
    sens = 0.0 if (conf["tp"] + conf["fn"]) == 0 else float(conf["tp"]) / float(conf["tp"] + conf["fn"])
    spec = 0.0 if (conf["tn"] + conf["fp"]) == 0 else float(conf["tn"]) / float(conf["tn"] + conf["fp"])
    return 0.5 * (sens + spec)


def _specificity(conf: Dict[str, int]) -> float:
    return 0.0 if (conf["tn"] + conf["fp"]) == 0 else float(conf["tn"]) / float(conf["tn"] + conf["fp"])


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    if np.unique(y_true).size < 2:
        return 0.5
    metric = tf.keras.metrics.AUC(curve="ROC")
    metric.update_state(y_true.astype(np.float32), np.asarray(y_prob).reshape(-1).astype(np.float32))
    return float(metric.result().numpy())


def _calibrate_decision_threshold(
    model: tf.keras.Model,
    cache_files_valid: List[str],
    slice_shape: Tuple[int, int, int],
) -> Dict[str, object]:
    """Select the volume-level threshold on the validation set.

    Candidates are 0.5 and every validation probability; the threshold with
    the highest balanced accuracy wins, ties broken by higher specificity.
    """
    y_true_list: List[float] = []
    y_prob_list: List[float] = []
    for cache_file in cache_files_valid:
        with h5py.File(cache_file, "r") as f:
            x = np.asarray(f["x"][:], dtype=np.float32)
            y_subject = float(np.asarray(f["y_subject"][:]).reshape(-1)[0])
        y_true_list.append(y_subject)
        y_prob_list.append(_predict_subject_probability(model, x, slice_shape))
    y_true = np.asarray(y_true_list, dtype=np.float32)
    y_prob = np.asarray(y_prob_list, dtype=np.float32)
    candidates = sorted({0.5, *[float(v) for v in y_prob.tolist()]})
    best_thr = 0.5
    best_conf = _confusion(y_true, y_prob, best_thr)
    best_bal = _balanced_accuracy(best_conf)
    best_spec = _specificity(best_conf)
    for thr in candidates:
        conf = _confusion(y_true, y_prob, thr)
        bal = _balanced_accuracy(conf)
        spec = _specificity(conf)
        if bal > best_bal + 1e-12 or (abs(bal - best_bal) <= 1e-12 and spec > best_spec + 1e-12):
            best_thr = float(thr)
            best_conf = conf
            best_bal = bal
            best_spec = spec
    return {
        "threshold": float(best_thr),
        "balanced_accuracy": float(best_bal),
        "specificity": float(best_spec),
        "confusion": best_conf,
        "auc": float(_auc(y_true, y_prob)),
        "n_subjects": int(y_true.size),
    }


def _validate_disjoint_subjects(train_cache_files: List[str], valid_cache_files: List[str]) -> None:
    def _subjects(paths: List[str]) -> set:
        out = set()
        for path in paths:
            with h5py.File(path, "r") as f:
                out.add(str(f.attrs.get("subject", "")))
        return out

    train_subjects = _subjects(train_cache_files)
    valid_subjects = _subjects(valid_cache_files)
    overlap = sorted(s for s in train_subjects.intersection(valid_subjects) if s)
    if overlap:
        raise ValueError(f"Data leakage detected: subjects appear in both train and valid splits: {overlap}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the ME-MoD motion detector.")
    parser.add_argument("--config", required=True, help="Path to the JSON config, e.g. configs/memod.json")
    parser.add_argument("--cache_root", default="cache/memod")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--out_dir", default=None, help="Run directory (default: runs/memod/<dataset>_<timestamp>)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _setup_gpu()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    train_cfg = config.get("train", {})
    classification_training = config.get("classification", {}).get("training", {})
    aggregation_cfg = dict(config.get("classification", {}).get("aggregation", {}))

    cache_files_train = build_cache_for_mode(config, mode="train", cache_root=args.cache_root, rebuild_cache=args.rebuild_cache)
    cache_files_valid = build_cache_for_mode(config, mode="valid", cache_root=args.cache_root, rebuild_cache=args.rebuild_cache)
    _validate_disjoint_subjects(cache_files_train, cache_files_valid)
    slice_shape = infer_sample_shape(cache_files_train)

    out_dir = abs_from_repo(args.out_dir) if args.out_dir else abs_from_repo(
        os.path.join(
            str(config["setting"].get("root_path", "runs/memod")),
            f"{config['setting']['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    )
    os.makedirs(out_dir, exist_ok=True)
    config_used_path = os.path.join(out_dir, "config_used.json")

    model = build_memod(
        slice_shape=slice_shape,
        model_cfg=config.get("model", {}),
        aggregation_cfg=aggregation_cfg,
    )
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    class_weight = (
        compute_subject_class_weights(cache_files_train)
        if str(config.get("classification", {}).get("class_balance", "none")).lower() == "weighted"
        else None
    )
    train_model = SubjectGateTrainer(base_model=model, cfg=classification_training, class_weight=class_weight)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    batch_size = int(train_cfg.get("subject_batch_size", train_cfg.get("batch_size", 2)))
    max_slices = train_cfg.get("max_slices", None)
    if max_slices is not None:
        max_slices = int(max_slices)
    train_ds = make_tf_subject_dataset(cache_files_train, batch_size=batch_size, shuffle=True, seed=args.seed, max_slices=max_slices, training=True)
    valid_ds = make_tf_subject_dataset(cache_files_valid, batch_size=batch_size, shuffle=False, seed=args.seed, max_slices=max_slices, training=False)
    steps_per_epoch = int(math.ceil(count_subject_samples(cache_files_train) / float(batch_size)))
    validation_steps = int(math.ceil(count_subject_samples(cache_files_valid) / float(batch_size)))

    monitor, monitor_mode = _resolve_monitor(train_cfg)
    config.setdefault("train", {})["monitor"] = monitor
    config["train"]["monitor_mode"] = monitor_mode
    save_json(config_used_path, config)
    callbacks = _build_callbacks(
        out_dir=out_dir,
        target_model=model,
        monitor=monitor,
        mode=monitor_mode,
        patience=int(train_cfg.get("early_stopping_patience", 50)),
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    history = train_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=int(train_cfg.get("train_epoch", 150)),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    threshold_info = _calibrate_decision_threshold(model, cache_files_valid, slice_shape)
    config.setdefault("classification", {})["decision_threshold"] = float(threshold_info["threshold"])
    save_json(os.path.join(out_dir, "decision_threshold.json"), threshold_info)
    save_json(config_used_path, config)

    model.save(os.path.join(out_dir, "final.keras"))
    model.save_weights(os.path.join(out_dir, "final_weights.h5"))
    save_json(
        os.path.join(out_dir, "history.json"),
        {k: [float(v) for v in vals] for k, vals in history.history.items()},
    )
    print(
        "[ME-MoD] calibrated threshold: "
        f"{threshold_info['threshold']:.4f} "
        f"(balanced_acc={threshold_info['balanced_accuracy']:.4f}, specificity={threshold_info['specificity']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
