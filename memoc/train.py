"""Train ME-MoC on paired motion-corrupted / motion-free multi-echo volumes.

Example (from the repository root):

    python -m memoc.train --config configs/memoc.json

Outputs go to ``<setting.root_path>/<timestamp>_<setting.save_folder>/``:
``model/`` (latest.h5, best-psnr.h5, best-ssim.h5, best-snr.h5, epoch-N.h5,
final.h5, training_config.json) and ``logs/`` (history.csv, TensorBoard).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from .cache_builder import build_cache_for_mode
from .dataset import count_records, infer_sample_shape, make_tf_dataset
from .models import build_memoc_from_config, compile_memoc

VARIANT = "attention_phys"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs_from_repo(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root(), path))


def _enable_gpu_settings(enable_mixed_precision: bool) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    if enable_mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception as e:
            print(f"[ME-MoC] mixed precision not enabled: {e}")


def _build_distribution_strategy(distribution: str = "auto") -> tf.distribute.Strategy:
    mode = str(distribution or "auto").strip().lower()
    if mode in ("off", "none", "single"):
        strategy = tf.distribute.get_strategy()
        print("[ME-MoC] distribution: single-device strategy")
        return strategy

    mirrored_copy_aliases = ("mirrored_copy", "mirrored_hierarchical", "mirrored_cpu_copy")
    if mode not in ("auto", "mirrored", *mirrored_copy_aliases):
        raise ValueError(
            f"Unsupported distribution mode: {distribution}. "
            "Expected auto|mirrored|mirrored_copy|single|off."
        )

    logical_gpus = tf.config.list_logical_devices("GPU")
    if len(logical_gpus) <= 1:
        strategy = tf.distribute.get_strategy()
        print(
            f"[ME-MoC] distribution: single-device strategy "
            f"(visible_logical_gpus={len(logical_gpus)})"
        )
        return strategy

    if mode in mirrored_copy_aliases:
        cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
        print(
            f"[ME-MoC] distribution: MirroredStrategy[HierarchicalCopyAllReduce] "
            f"(replicas={strategy.num_replicas_in_sync}, visible_logical_gpus={len(logical_gpus)})"
        )
        return strategy

    strategy = tf.distribute.MirroredStrategy()
    print(
        f"[ME-MoC] distribution: MirroredStrategy[NCCL/default] "
        f"(replicas={strategy.num_replicas_in_sync}, visible_logical_gpus={len(logical_gpus)})"
    )
    return strategy


def _validate_config(config: Dict, input_shape: Tuple[int, ...]) -> None:
    dataset_name = config.get("setting", {}).get("dataset", "mri_3decho")
    if dataset_name != "mri_3decho":
        raise ValueError(f"ME-MoC supports setting.dataset='mri_3decho' only, got {dataset_name!r}.")
    train_cfg = config.get("train", {})
    variant = str(train_cfg.get("variant", VARIANT)).lower()
    if variant != VARIANT:
        raise ValueError(f"This repository provides the '{VARIANT}' ME-MoC model only; config requests {variant!r}.")

    num_echoes = int(input_shape[2]) if len(input_shape) >= 3 else None
    physics_cfg = train_cfg.get("physics", {})
    lambda_phys = float(train_cfg.get("lambda_phys", 0.0))
    use_te_weighting = bool(physics_cfg.get("use_te_weighting", False))
    te_ms = physics_cfg.get("te_ms", None)

    if lambda_phys <= 0.0:
        print(f"[ME-MoC] note: train.lambda_phys={lambda_phys:.4f}, so the T2*-consistency term is disabled.")
        return

    if te_ms is None:
        if use_te_weighting:
            raise ValueError(
                "[ME-MoC] train.physics.te_ms is required when lambda_phys > 0 and "
                "use_te_weighting=True. Provide the echo times, or set lambda_phys=0, "
                "or disable use_te_weighting."
            )
        print(
            f"[ME-MoC] physics_config: lambda_phys={lambda_phys:.4f} using the unweighted "
            "second difference along the echo axis (assumes uniform echo spacing)."
        )
        return

    if not isinstance(te_ms, (list, tuple)):
        raise ValueError("[ME-MoC] train.physics.te_ms must be a list of echo times in milliseconds.")
    try:
        te_values = [float(t) for t in te_ms]
    except (TypeError, ValueError) as exc:
        raise ValueError("[ME-MoC] train.physics.te_ms contains a non-numeric echo time.") from exc
    if len(te_values) < 3:
        raise ValueError("[ME-MoC] train.physics.te_ms must contain at least 3 echo times.")
    if any(curr <= prev for prev, curr in zip(te_values[:-1], te_values[1:])):
        raise ValueError("[ME-MoC] train.physics.te_ms must be strictly increasing.")
    if num_echoes is not None and len(te_values) != num_echoes:
        raise ValueError(
            f"[ME-MoC] train.physics.te_ms length ({len(te_values)}) does not match "
            f"the input echo count ({num_echoes})."
        )
    print(
        f"[ME-MoC] physics_config: lambda_phys={lambda_phys:.4f} "
        f"use_te_weighting={use_te_weighting} te_ms={te_values}"
    )


def _print_runtime_debug(
    config: Dict,
    input_shape: Tuple[int, ...],
    batch_size: int,
    cache_root: str,
    strategy: tf.distribute.Strategy,
) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    policy = tf.keras.mixed_precision.global_policy()

    print(
        f"[ME-MoC] runtime: input_shape={tuple(input_shape)} batch_size={batch_size} "
        f"cuda_visible_devices={cuda_visible} "
        f"physical_gpus={len(physical_gpus)} logical_gpus={len(logical_gpus)} "
        f"mixed_precision_policy={policy.name} cache_root={cache_root} "
        f"strategy={strategy.__class__.__name__} replicas={int(strategy.num_replicas_in_sync)}"
    )
    if physical_gpus:
        gpu_names = ", ".join(getattr(gpu, "name", str(gpu)) for gpu in physical_gpus)
        print(f"[ME-MoC] visible_gpu_devices: {gpu_names}")

    p = config["method"]["unet3d"]
    train_cfg = config.get("train", {})
    print(
        f"[ME-MoC] model_config: filters_root={int(p.get('filters_root', 32))} "
        f"conv_times={int(p.get('conv_times', 3))} "
        f"up_down_times={int(p.get('up_down_times', 4))} "
        f"attn_heads={int(train_cfg.get('attn_heads', 2))} "
        f"attn_key_dim={int(train_cfg.get('attn_key_dim', 16))} "
        f"identity_ratio={float(train_cfg.get('identity_ratio', 0.0)):.3f}"
    )


class RuntimeTracker(tf.keras.callbacks.Callback):
    """Records per-epoch wall time in ``logs`` so CSVLogger writes it to history.csv."""

    def __init__(self, batch_size: int, steps_per_epoch: int):
        super().__init__()
        self.batch_size = max(1, int(batch_size))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self._epoch_start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = float(time.perf_counter() - self._epoch_start)
        step_ms = 1000.0 * epoch_time / float(self.steps_per_epoch)
        samples_per_sec = float(self.steps_per_epoch * self.batch_size) / max(epoch_time, 1e-6)
        logs["epoch_time_sec"] = epoch_time
        logs["step_time_ms"] = step_ms
        logs["samples_per_sec"] = samples_per_sec
        print(
            f"[ME-MoC] epoch_runtime: epoch={epoch+1} sec={epoch_time:.2f} "
            f"step_ms={step_ms:.2f} samples_per_sec={samples_per_sec:.2f}"
        )


class CleanIdentityEvalCallback(tf.keras.callbacks.Callback):
    """Evaluates the model on motion-free validation inputs (target = input).

    The resulting ``val_clean_*`` metrics measure how much the corrector
    changes data that does not need correction.
    """

    def __init__(self, base_model: tf.keras.Model, clean_ds: tf.data.Dataset):
        super().__init__()
        self.base_model = base_model
        self.clean_ds = clean_ds

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        results = self.base_model.evaluate(self.clean_ds, verbose=0, return_dict=True)
        for key, value in results.items():
            if key == "loss":
                logs["val_clean_loss"] = float(value)
            else:
                logs[f"val_clean_{key}"] = float(value)
        psnr = logs.get("val_clean_psnr")
        ssim = logs.get("val_clean_ssim")
        if psnr is not None or ssim is not None:
            print(
                f"[ME-MoC] clean_identity: epoch={epoch+1} "
                f"psnr={float(psnr or 0.0):.4f} ssim={float(ssim or 0.0):.4f}"
            )


class FirstEpochFiniteWarning(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if int(epoch) != 0:
            return
        logs = logs or {}
        non_finite = {}
        for key, value in logs.items():
            try:
                value_f = float(value)
            except Exception:
                continue
            if not math.isfinite(value_f):
                non_finite[key] = value_f
        if non_finite:
            print(f"[ME-MoC] warning: first epoch produced non-finite metrics. values={non_finite}")


class CheckpointSelector(tf.keras.callbacks.Callback):
    """Saves latest/epoch/best-{psnr,ssim,snr} weights and an optional deployable copy.

    ``best-psnr.h5`` (highest validation PSNR) is the checkpoint released with
    the paper. At the end of training it is copied to ``deployable.h5`` if it
    meets the optional thresholds in ``train.deployment_guard``.
    """

    def __init__(
        self,
        base_model: tf.keras.Model,
        ckpt_dir: str,
        save_epoch: int,
        baseline_psnr: Optional[float],
        psnr_tolerance: float,
        fallback_weights: Optional[str],
        min_val_psnr: float = 0.0,
        min_val_clean_psnr: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.ckpt_dir = ckpt_dir
        self.save_epoch = int(save_epoch)
        self.baseline_psnr = baseline_psnr
        self.psnr_tolerance = float(psnr_tolerance)
        self.fallback_weights = fallback_weights
        self.min_val_psnr = float(min_val_psnr)
        self.min_val_clean_psnr = float(min_val_clean_psnr)
        self.best = {"psnr": -float("inf"), "ssim": -float("inf"), "snr": -float("inf")}
        self.best_clean_psnr = -float("inf")
        self.best_psnr_path = os.path.join(self.ckpt_dir, "best-psnr.h5")
        self.deploy_path = os.path.join(self.ckpt_dir, "deployable.h5")
        self.state_path = os.path.join(self.ckpt_dir, "checkpoint_state.json")
        self._restore_state()

    def _restore_state(self) -> None:
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)
            stored_best = state.get("best", {})
            for tag in self.best:
                value = float(stored_best.get(tag, self.best[tag]))
                if math.isfinite(value):
                    self.best[tag] = value
            clean_value = float(state.get("best_clean_psnr", self.best_clean_psnr))
            if math.isfinite(clean_value):
                self.best_clean_psnr = clean_value
            print(
                "[ME-MoC] restored checkpoint-selection state: "
                f"last_completed_epoch={state.get('last_completed_epoch', 'unknown')}, "
                f"best={self.best}, best_clean_psnr={self.best_clean_psnr}"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[ME-MoC] warning: could not restore {self.state_path}: {exc}")

    def _save_state(self, epoch_num: int) -> None:
        state = {
            "last_completed_epoch": int(epoch_num),
            "best": {tag: float(value) for tag, value in self.best.items()},
            "best_clean_psnr": float(self.best_clean_psnr),
        }
        temp_path = f"{self.state_path}.tmp"
        with open(temp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_path, self.state_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        epoch_num = int(epoch) + 1
        self.base_model.save_weights(os.path.join(self.ckpt_dir, "latest.h5"))
        if (
            (self.save_epoch > 0 and (epoch_num % self.save_epoch == 0))
            or ((epoch_num % 10 == 0) and (epoch_num <= 100))
            or (epoch_num <= 10)
        ):
            self.base_model.save_weights(os.path.join(self.ckpt_dir, f"epoch-{epoch_num}.h5"))

        for tag in ("psnr", "ssim", "snr"):
            val = logs.get(f"val_{tag}")
            if val is None:
                continue
            val_f = float(val)
            if not math.isfinite(val_f):
                print(f"[ME-MoC] warning: skipping non-finite val_{tag}={val_f} for checkpoint selection.")
                continue
            if val_f > self.best[tag]:
                self.best[tag] = val_f
                self.base_model.save_weights(os.path.join(self.ckpt_dir, f"best-{tag}.h5"))

        clean_psnr = logs.get("val_clean_psnr")
        if clean_psnr is not None:
            clean_psnr_f = float(clean_psnr)
            if math.isfinite(clean_psnr_f):
                self.best_clean_psnr = max(self.best_clean_psnr, clean_psnr_f)
            else:
                print(f"[ME-MoC] warning: skipping non-finite val_clean_psnr={clean_psnr_f}.")
        self._save_state(epoch_num)

    def on_train_end(self, logs=None):
        best_psnr = float(self.best["psnr"])
        guard_active = self.baseline_psnr is not None
        has_finite_psnr = math.isfinite(best_psnr) and best_psnr > -float("inf")
        threshold = None
        passed = has_finite_psnr and best_psnr >= float(self.min_val_psnr)
        clean_guard_active = self.best_clean_psnr > -float("inf") or self.min_val_clean_psnr > 0.0
        if clean_guard_active:
            passed = passed and math.isfinite(float(self.best_clean_psnr)) and float(self.best_clean_psnr) >= float(self.min_val_clean_psnr)
        if guard_active:
            threshold = float(self.baseline_psnr) - float(self.psnr_tolerance)
            passed = passed and best_psnr >= threshold

        if passed and os.path.exists(self.best_psnr_path):
            shutil.copy2(self.best_psnr_path, self.deploy_path)
            print(f"[ME-MoC] deployment check passed. best_val_psnr={best_psnr:.4f}. deployable={self.deploy_path}")
            return

        if self.fallback_weights and os.path.exists(self.fallback_weights):
            shutil.copy2(self.fallback_weights, self.deploy_path)
            print(
                f"[ME-MoC] deployment check failed. "
                f"best_val_psnr={best_psnr:.4f}, best_val_clean_psnr={self.best_clean_psnr:.4f}, "
                f"threshold={threshold}, min_val_psnr={self.min_val_psnr:.4f}, "
                f"min_val_clean_psnr={self.min_val_clean_psnr:.4f}. "
                f"Using fallback weights: {self.fallback_weights}"
            )
            return

        print(
            "[ME-MoC] deployment check failed. No deployable weights exported. "
            f"best_val_psnr={best_psnr:.4f}, best_val_clean_psnr={self.best_clean_psnr:.4f}, "
            f"threshold={threshold}, min_val_psnr={self.min_val_psnr:.4f}, "
            f"min_val_clean_psnr={self.min_val_clean_psnr:.4f}."
        )


def _save_training_config(config: Dict, out_dir: str, runtime: Dict) -> None:
    saved_config = json.loads(json.dumps(config))
    saved_config.setdefault("train", {})
    saved_config["train"]["variant"] = VARIANT
    saved_config.setdefault("_runtime", {})
    saved_config["_runtime"].update(runtime)

    out_path = os.path.join(out_dir, "training_config.json")
    with open(out_path, "w") as f:
        json.dump(saved_config, f, indent=2)
    print(f"[ME-MoC] saved training config to {out_path}")


def _build_model_and_compile(config: Dict, input_shape: Tuple[int, ...]) -> tf.keras.Model:
    train_cfg = config.get("train", {})
    physics_cfg = train_cfg.get("physics", {})
    model = build_memoc_from_config(config, input_shape=input_shape)
    compile_memoc(
        model=model,
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        loss_type=str(train_cfg.get("lossType", "l2")),
        lambda_mag=float(train_cfg.get("lambda_mag", 0.0)),
        lambda_phys=float(train_cfg.get("lambda_phys", 0.0)),
        te_ms=physics_cfg.get("te_ms", None),
        use_te_weighting=bool(physics_cfg.get("use_te_weighting", False)),
        physics_mask_mode=str(physics_cfg.get("mask_source", "echo1_true")).lower(),
        physics_mask_threshold=float(physics_cfg.get("mask_threshold", 0.03)),
        psnr_max_val=float(train_cfg.get("psnr_max_val", 1.0)),
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the ME-MoC motion-correction model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config, e.g. configs/memoc.json")
    parser.add_argument("--cache_root", default="cache/memoc")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument(
        "--distribution",
        default="auto",
        help="TensorFlow distribution mode: auto|mirrored|mirrored_copy|single|off. "
             "auto uses MirroredStrategy when multiple GPUs are visible; "
             "mirrored_copy uses HierarchicalCopyAllReduce instead of NCCL.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override train.train_epoch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume_run",
        default=None,
        help="Resume an interrupted run from its run directory (must contain training_backup/).",
    )
    parser.add_argument(
        "--initial_weights",
        default=None,
        help="Initialize from a weights file without optimizer state (e.g. for fine-tuning).",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=0,
        help="Completed epoch count used with --initial_weights (default: 0).",
    )
    args = parser.parse_args()
    if args.resume_run and args.initial_weights:
        parser.error("--resume_run and --initial_weights are mutually exclusive.")
    if args.resume_run and int(args.initial_epoch) != 0:
        parser.error("--initial_epoch cannot be combined with --resume_run; the backup stores the epoch.")
    if args.resume_run and args.rebuild_cache:
        parser.error("--rebuild_cache cannot be combined with --resume_run; resume with the unchanged cache.")
    if int(args.initial_epoch) < 0:
        parser.error("--initial_epoch must be zero or greater.")

    with open(_abs_from_repo(args.config), "r") as f:
        config = json.load(f)

    tf.keras.utils.set_random_seed(int(args.seed))
    _enable_gpu_settings(enable_mixed_precision=args.mixed_precision)
    strategy = _build_distribution_strategy(args.distribution)

    train_cfg = config.get("train", {})
    physics_cfg = train_cfg.get("physics", {})

    cache_root = _abs_from_repo(args.cache_root)
    print(f"[ME-MoC] cache_root={cache_root}")
    if bool(args.mixed_precision):
        print(
            "[ME-MoC] warning: mixed precision is enabled. The released model was "
            "trained in float32; validate stability before relying on mixed precision."
        )
    train_cache = build_cache_for_mode(config, mode="train", cache_root=cache_root, rebuild_cache=args.rebuild_cache)
    valid_cache = build_cache_for_mode(config, mode="valid", cache_root=cache_root, rebuild_cache=args.rebuild_cache)

    batch_size = int(config["train"]["batch_size"])
    if int(strategy.num_replicas_in_sync) > 1 and (batch_size % int(strategy.num_replicas_in_sync)) != 0:
        print(
            f"[ME-MoC] warning: global batch_size={batch_size} is not divisible by "
            f"replicas={int(strategy.num_replicas_in_sync)}."
        )
    train_ds = make_tf_dataset(train_cache, batch_size=batch_size, shuffle=True, seed=int(args.seed))
    valid_ds = make_tf_dataset(valid_cache, batch_size=batch_size, shuffle=False, seed=int(args.seed))
    # Motion-free validation slices fed as input with themselves as target.
    clean_valid_ds = valid_ds.map(lambda _x, y: (y, y), num_parallel_calls=tf.data.AUTOTUNE)

    identity_ratio = float(train_cfg.get("identity_ratio", 0.0))
    if not 0.0 <= identity_ratio <= 1.0:
        raise ValueError(f"train.identity_ratio must be within [0, 1], got {identity_ratio}.")
    if identity_ratio > 0.0:
        # Replace a fraction of training inputs by their motion-free targets.
        def _inject_identity_batch(x, y):
            selector = (
                tf.random.uniform([tf.shape(x)[0], 1, 1, 1, 1], dtype=tf.float32)
                < float(identity_ratio)
            )
            return tf.where(selector, y, x), y

        train_ds = train_ds.map(_inject_identity_batch, num_parallel_calls=tf.data.AUTOTUNE)

    input_shape = infer_sample_shape(train_cache)
    _validate_config(config, input_shape=input_shape)
    _print_runtime_debug(
        config=config,
        input_shape=input_shape,
        batch_size=batch_size,
        cache_root=cache_root,
        strategy=strategy,
    )
    with strategy.scope():
        model = _build_model_and_compile(config, input_shape=input_shape)
    initial_weights_path = None
    if args.initial_weights:
        initial_weights_path = _abs_from_repo(str(args.initial_weights))
        if not os.path.isfile(initial_weights_path):
            raise FileNotFoundError(f"Initial weights do not exist: {initial_weights_path}")
        model.load_weights(initial_weights_path)
        print(
            f"[ME-MoC] loaded weights-only checkpoint: {initial_weights_path}. "
            "Optimizer state is new, so this is not an exact continuation."
        )
    model.summary()

    root_path = _abs_from_repo(config["setting"].get("root_path", "runs/memoc"))
    save_folder = config["setting"].get("save_folder", "ME-MoC")
    resume_mode = bool(args.resume_run)
    if resume_mode:
        run_dir = _abs_from_repo(str(args.resume_run))
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    else:
        run_dir = os.path.join(root_path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{save_folder}")
    ckpt_dir = os.path.join(run_dir, "model")
    log_dir = os.path.join(run_dir, "logs")
    backup_dir = os.path.join(run_dir, "training_backup")
    backup_has_files = (
        os.path.isdir(backup_dir)
        and any(filenames for _root, _dirs, filenames in os.walk(backup_dir))
    )
    if resume_mode and not backup_has_files:
        raise FileNotFoundError(
            f"No resumable training backup was found at {backup_dir}. "
            "Use --initial_weights RUN_DIR/model/latest.h5 --initial_epoch COMPLETED_EPOCHS instead."
        )
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    runtime = {
        "source_config": os.path.abspath(args.config),
        "cache_root": cache_root,
        "rebuild_cache": bool(args.rebuild_cache),
        "mixed_precision": bool(args.mixed_precision),
        "distribution": str(args.distribution),
        "num_replicas_in_sync": int(strategy.num_replicas_in_sync),
        "epochs_override": args.epochs,
        "seed": int(args.seed),
        "resume_run": run_dir if resume_mode else None,
        "initial_weights": initial_weights_path,
        "initial_epoch": int(args.initial_epoch),
    }
    if resume_mode:
        resume_log_path = os.path.join(log_dir, "resume_invocations.jsonl")
        with open(resume_log_path, "a") as f:
            f.write(json.dumps({"timestamp": datetime.now().isoformat(), **runtime}) + "\n")
        print(f"[ME-MoC] resuming existing run: {run_dir}")
    else:
        _save_training_config(config=config, out_dir=ckpt_dir, runtime=runtime)

    train_records = count_records(train_cache)
    valid_records = count_records(valid_cache)
    steps_per_epoch = int(math.ceil(train_records / float(batch_size)))
    validation_steps = int(math.ceil(valid_records / float(batch_size)))

    guard_cfg = train_cfg.get("deployment_guard", {})
    baseline_psnr = guard_cfg.get("baseline_val_psnr")
    fallback_weights = guard_cfg.get("fallback_weights", None)
    if fallback_weights:
        fallback_weights = _abs_from_repo(str(fallback_weights))

    checkpoint = CheckpointSelector(
        base_model=model,
        ckpt_dir=ckpt_dir,
        save_epoch=int(train_cfg.get("save_epoch", 10)),
        baseline_psnr=float(baseline_psnr) if baseline_psnr is not None else None,
        psnr_tolerance=float(guard_cfg.get("psnr_tolerance", 0.05)),
        fallback_weights=fallback_weights,
        min_val_psnr=float(guard_cfg.get("min_val_psnr", 0.0)),
        min_val_clean_psnr=float(guard_cfg.get("min_val_clean_psnr", 0.0)),
    )
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=backup_dir,
            save_freq="epoch",
            delete_checkpoint=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        FirstEpochFiniteWarning(),
        RuntimeTracker(batch_size=batch_size, steps_per_epoch=steps_per_epoch),
        CleanIdentityEvalCallback(base_model=model, clean_ds=clean_valid_ds),
        checkpoint,
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="epoch"),
        tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "history.csv"), append=resume_mode),
    ]

    epochs = int(args.epochs) if args.epochs is not None else int(train_cfg.get("train_epoch", 100))
    if int(args.initial_epoch) >= epochs:
        raise ValueError(
            f"--initial_epoch ({args.initial_epoch}) must be smaller than total epochs ({epochs})."
        )
    print(
        f"[ME-MoC] training: epochs={epochs} initial_epoch={int(args.initial_epoch)} "
        f"steps_per_epoch={steps_per_epoch} batch_size={batch_size}"
    )

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        initial_epoch=int(args.initial_epoch),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(ckpt_dir, "final.h5")
    model.save_weights(final_path)

    best_val_psnr = float(checkpoint.best["psnr"]) if math.isfinite(float(checkpoint.best["psnr"])) else None
    best_val_clean_psnr = (
        float(checkpoint.best_clean_psnr) if math.isfinite(float(checkpoint.best_clean_psnr)) else None
    )
    summary = {
        "variant": VARIANT,
        "physics_mask_source": str(physics_cfg.get("mask_source", "echo1_true")).lower(),
        "physics_use_te_weighting": bool(physics_cfg.get("use_te_weighting", False)),
        "identity_ratio": identity_ratio,
        "epochs": epochs,
        "initial_epoch": int(args.initial_epoch),
        "resumed": resume_mode,
        "resume_run": run_dir if resume_mode else None,
        "initial_weights": runtime["initial_weights"],
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "train_records": train_records,
        "valid_records": valid_records,
        "best_val_psnr": best_val_psnr,
        "best_val_clean_psnr": best_val_clean_psnr,
        "run_dir": run_dir,
        "final_weights": final_path,
        "deployable_weights": os.path.join(ckpt_dir, "deployable.h5"),
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ME-MoC] done. outputs in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
