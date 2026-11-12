"""ME-MoC: complex-valued (2+1)D U-Net with echo-axis attention.

This module defines the multi-echo motion corrector used in MDCC:

  * complex-valued convolutions, transposed convolutions, instance
    normalisation and CReLU, so real and imaginary parts stay coupled;
  * a factorized (2+1)D block (3x3x1 spatial conv, then 1x1x3 echo conv);
  * multi-head self-attention along the echo axis at the bottleneck;
  * a combined loss with complex, magnitude and T2*-consistency terms.

Tensor convention: a feature map has shape (B, H, W, E, 2F). The first F
channels hold the real parts and the last F channels the imaginary parts
of F complex feature maps. Network input and output are (B, H, W, E, 2).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Conv3D,
    Conv3DTranspose,
    Input,
    Layer,
    MaxPool3D,
    MultiHeadAttention,
    Subtract,
)
from tensorflow.keras.models import Model


# ---------------------------------------------------------------------------
# Complex-valued building blocks
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="MEMoC")
class ComplexConv3D(Layer):
    """Complex-valued 3D convolution.

    Complex multiplication rule applied per filter:
        out_r = W_r * x_r - W_i * x_i
        out_i = W_r * x_i + W_i * x_r

    W_r and W_i are two independent real-valued Conv3D layers that together
    parameterise a complex weight tensor W = W_r + i W_i.
    """

    def __init__(self, filters: int, kernel_size, padding: str = "same",
                 use_bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = use_bias
        self.conv_r = Conv3D(filters, kernel_size, padding=padding,
                             use_bias=use_bias, name="Wr")
        self.conv_i = Conv3D(filters, kernel_size, padding=padding,
                             use_bias=use_bias, name="Wi")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return tf.concat([out_r, out_i], axis=-1)  # (B,H,W,E, 2*filters)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "kernel_size": self.kernel_size,
                    "padding": self.padding, "use_bias": self.use_bias})
        return cfg


@tf.keras.utils.register_keras_serializable(package="MEMoC")
class ComplexConv3DTranspose(Layer):
    """Complex-valued transposed 3D convolution (upsampling).

    Mirrors ComplexConv3D: two real Conv3DTranspose layers (Wr, Wi) apply
        out_r = Wr^T * x_r - Wi^T * x_i
        out_i = Wr^T * x_i + Wi^T * x_r
    so the phase relationship is preserved in the decoder path.
    """

    def __init__(self, filters: int, kernel_size, padding: str = "same",
                 strides=(2, 2, 1), use_bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.use_bias = use_bias
        self.conv_r = Conv3DTranspose(filters, kernel_size, padding=padding,
                                      strides=strides, use_bias=use_bias, name="Wr")
        self.conv_i = Conv3DTranspose(filters, kernel_size, padding=padding,
                                      strides=strides, use_bias=use_bias, name="Wi")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return tf.concat([out_r, out_i], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "kernel_size": self.kernel_size,
                    "padding": self.padding, "strides": self.strides,
                    "use_bias": self.use_bias})
        return cfg


@tf.keras.utils.register_keras_serializable(package="MEMoC")
class ComplexInstanceNorm(Layer):
    """Instance normalisation for complex feature maps (B, H, W, E, 2F).

    Real and imaginary parts are divided by the same per-channel mean
    magnitude, which preserves their ratio (the phase). A learnable gain
    gamma is applied per complex channel.
    """

    def __init__(self, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        C = int(input_shape[-1])
        if C % 2 != 0:
            raise ValueError(
                f"ComplexInstanceNorm expects even last dim (2*F), got {C}."
            )
        F = C // 2  # number of complex channels
        self.gamma = self.add_weight(name="gamma", shape=(F,), initializer="ones")
        # beta is kept non-trainable and unused in the forward pass so the
        # released checkpoint keeps its weight layout; a real-only bias would
        # break the symmetry between real and imaginary branches.
        self.beta = self.add_weight(
            name="beta",
            shape=(F,),
            initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        mag = tf.sqrt(tf.square(x_r) + tf.square(x_i) + self.epsilon)
        # Normalise over spatial and echo axes (1,2,3), per sample and channel.
        axes = list(range(1, len(x.shape) - 1))
        mean_mag = tf.reduce_mean(mag, axis=axes, keepdims=True)
        scale = mean_mag + self.epsilon
        gamma = tf.cast(self.gamma, x.dtype)
        x_r_n = (x_r / scale) * gamma
        x_i_n = (x_i / scale) * gamma
        return tf.concat([x_r_n, x_i_n], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg["epsilon"] = self.epsilon
        return cfg


@tf.keras.utils.register_keras_serializable(package="MEMoC")
class ComplexReLU(Layer):
    """CReLU activation: ReLU applied independently to real and imaginary parts."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_r, x_i = tf.split(x, num_or_size_splits=2, axis=-1)
        return tf.concat([tf.nn.relu(x_r), tf.nn.relu(x_i)], axis=-1)

    def get_config(self):
        return super().get_config()


def _complex_factorized_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """One (2+1)D complex factorized block (FBlock):
        ComplexConv3D(3x3x1) -> ComplexInstanceNorm -> CReLU
        ComplexConv3D(1x1x3) -> ComplexInstanceNorm -> CReLU
    """
    # Spatial 3x3x1: in-plane context within each echo.
    x = ComplexConv3D(filters, kernel_size=(3, 3, 1), name=f"{name}/SpatialConv")(x)
    x = ComplexInstanceNorm(name=f"{name}/SpatialNorm")(x)
    x = ComplexReLU(name=f"{name}/SpatialAct")(x)
    # Echo 1x1x3: mixes adjacent echo times at each spatial position.
    x = ComplexConv3D(filters, kernel_size=(1, 1, 3), name=f"{name}/EchoConv")(x)
    x = ComplexInstanceNorm(name=f"{name}/EchoNorm")(x)
    x = ComplexReLU(name=f"{name}/EchoAct")(x)
    return x


# ---------------------------------------------------------------------------
# Echo-axis multi-head self-attention
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="MEMoC")
class EchoAxisAttention(Layer):
    """Multi-head self-attention across the echo-time dimension.

    At each spatial position (h, w) the E echo feature vectors form a
    sequence of length E, and pre-norm scaled dot-product attention with a
    residual connection is applied to that sequence. The layer is used at
    the bottleneck, where H and W are small (16x12 for a 256x192 input with
    four down-sampling levels).

    Parameters
    ----------
    num_echoes : int
        E, the length of the echo sequence.
    num_heads : int
        Number of attention heads.
    key_dim : int
        Dimension of the Q/K projections per head.
    """

    def __init__(self, num_echoes: int, num_heads: int = 2,
                 key_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_echoes = num_echoes
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            attention_axes=None, name="mha"
        )
        self.epsilon = 1e-5

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.ln_gamma = self.add_weight(name="ln_gamma", shape=(c,), initializer="ones")
        self.ln_beta = self.add_weight(name="ln_beta", shape=(c,), initializer="zeros")
        super().build(input_shape)

    def _manual_layer_norm(self, x: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x_hat = (x - mean) * tf.math.rsqrt(var + tf.cast(self.epsilon, x.dtype))
        return x_hat * tf.cast(self.ln_gamma, x.dtype) + tf.cast(self.ln_beta, x.dtype)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: (B, H, W, E, C). B/H/W/E are dynamic; C must be static so that
        # MultiHeadAttention can build its Q/K/V projections at graph build time.
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        E = tf.shape(x)[3]
        C = x.shape[-1]
        # Collapse batch and spatial dims: (B*H*W, E, C).
        x_flat = tf.reshape(x, [B * H * W, E, C])
        # Pre-norm -> attention -> residual.
        x_norm = self._manual_layer_norm(x_flat)
        attended = self.mha(x_norm, x_norm, training=training)
        x_out = x_flat + attended
        return tf.reshape(x_out, [B, H, W, E, C])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_echoes": self.num_echoes, "num_heads": self.num_heads,
                    "key_dim": self.key_dim})
        return cfg


# ---------------------------------------------------------------------------
# Loss and monitoring metrics
# ---------------------------------------------------------------------------

def img_output_magnitude(t: tf.Tensor) -> tf.Tensor:
    """Magnitude of a complex image output with channel layout [..., Re, Im]."""
    real, imag = tf.split(t, num_or_size_splits=2, axis=-1)
    return tf.squeeze(tf.sqrt(tf.square(real) + tf.square(imag) + 1e-8), axis=-1)


def img_phase_cosine_error(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    mask_threshold: float = 0.03,
) -> tf.Tensor:
    """Foreground-masked phase cosine error (1 - cos(dphi)) on complex outputs."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tr, ti = tf.split(y_true, num_or_size_splits=2, axis=-1)
    pr, pi = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    tmag = tf.sqrt(tf.square(tr) + tf.square(ti) + 1e-8)
    pmag = tf.sqrt(tf.square(pr) + tf.square(pi) + 1e-8)
    cos = (tr * pr + ti * pi) / (tmag * pmag + 1e-8)
    echo1 = tmag[..., 0:1, :]
    scale = tf.reduce_max(echo1, axis=(1, 2), keepdims=True)
    base_mask = tf.cast(echo1 >= tf.maximum(scale * float(mask_threshold), 1e-8), cos.dtype)
    mask = tf.repeat(base_mask, repeats=tf.shape(cos)[3], axis=3)
    return tf.reduce_sum((1.0 - cos) * mask) / (tf.reduce_sum(mask) + 1e-8)


def make_img_loss(
    lambda_mag: float = 0.2,
    lambda_phys: float = 0.02,
    loss_type: str = "l2",
    te_ms: Optional[List[float]] = None,
    use_te_weighting: bool = False,
    mask_mode: str = "echo1_true",
    mask_threshold: float = 0.03,
) -> Callable:
    """Combined loss for complex multi-echo MRI correction.

        L = (1 - lambda_m - lambda_p) * L_complex + lambda_m * L_mag + lambda_p * L_phys

      L_complex : penalty on the raw (Re, Im) channels
      L_mag     : penalty on the magnitudes |z|
      L_phys    : T2* mono-exponential consistency of the prediction. For
                  mono-exponential decay log|S(TE)| is linear in TE, so the
                  second finite difference along the echo axis should vanish:
                  L_phys = mean((d^2 log|S_pred|)^2) over the foreground mask.

    Parameters
    ----------
    lambda_mag       : weight of the magnitude term
    lambda_phys      : weight of the T2*-consistency term
    loss_type        : reconstruction penalty, ``l1`` or ``l2``
    te_ms            : echo times in ms; required when use_te_weighting=True
    use_te_weighting : divide differences by the echo spacing (non-uniform TE)
    mask_mode        : foreground mask for L_phys: none | echo1_true | echo1_pred
    mask_threshold   : mask threshold as a fraction of the slice echo-1 maximum
    """
    lam_m = float(lambda_mag)
    lam_p = float(lambda_phys)
    lam_c = 1.0 - lam_m - lam_p
    recon_loss_type = str(loss_type).strip().lower()
    if recon_loss_type not in ("l1", "l2"):
        raise ValueError(
            f"Unsupported loss_type={loss_type!r}. Expected 'l1' or 'l2'."
        )
    te = None

    if te_ms is not None:
        try:
            te_values = [float(t) for t in te_ms]
        except (TypeError, ValueError) as exc:
            raise ValueError("make_img_loss received a non-numeric train.physics.te_ms value.") from exc
        if len(te_values) < 3:
            raise ValueError("make_img_loss requires at least 3 echo times when train.physics.te_ms is provided.")
        if any(curr <= prev for prev, curr in zip(te_values[:-1], te_values[1:])):
            raise ValueError("make_img_loss requires train.physics.te_ms to be strictly increasing.")
        te = tf.constant([t / 1e3 for t in te_values], dtype=tf.float32)
    elif lam_p > 0.0 and use_te_weighting:
        raise ValueError(
            "make_img_loss requires explicit train.physics.te_ms when lambda_phys > 0 "
            "and use_te_weighting=True. Provide the true echo times, or set "
            "lambda_phys=0, or disable use_te_weighting."
        )

    if lam_c < 0.0:
        import warnings
        warnings.warn(
            f"make_img_loss: lambda_mag={lam_m} + lambda_phys={lam_p} = {lam_m+lam_p:.2f} > 1.0. "
            "The complex-domain term weight is clamped to 0. "
            "Consider reducing lambda_mag or lambda_phys so they sum to <= 1.",
            stacklevel=2,
        )
        lam_c = 0.0
    mode = str(mask_mode).lower()
    if mode not in ("none", "echo1_true", "echo1_pred"):
        raise ValueError(f"Unsupported mask_mode={mask_mode}. Expected none|echo1_true|echo1_pred.")

    def _mag(t: tf.Tensor) -> tf.Tensor:
        """Magnitude from the (..., 2F) complex channel convention."""
        t = tf.cast(t, tf.float32)
        F = t.shape[-1] // 2
        return tf.sqrt(tf.square(t[..., :F]) + tf.square(t[..., F:]) + 1e-8)

    def _reconstruction_penalty(error: tf.Tensor) -> tf.Tensor:
        if recon_loss_type == "l1":
            return tf.reduce_mean(tf.abs(error))
        return tf.reduce_mean(tf.square(error))

    def _build_physics_mask(y_true: tf.Tensor, y_pred: tf.Tensor) -> Optional[tf.Tensor]:
        if mode == "none":
            return None
        if mode == "echo1_true":
            ref = tf.squeeze(_mag(y_true), axis=-1)
        else:
            ref = tf.squeeze(_mag(y_pred), axis=-1)

        e1 = ref[..., 0:1]
        scale = tf.reduce_max(e1, axis=(1, 2), keepdims=True)
        thresh = tf.maximum(scale * float(mask_threshold), 1e-8)
        mask2d = tf.cast(e1 >= thresh, tf.float32)
        num_echo = tf.shape(ref)[-1]
        return tf.repeat(mask2d, repeats=num_echo, axis=-1)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        l_c = _reconstruction_penalty(y_pred - y_true)
        l_m = _reconstruction_penalty(_mag(y_pred) - _mag(y_true))

        if lam_p > 0.0:
            # y_pred: (B, H, W, E, 2); magnitude per echo: (B, H, W, E)
            mag_pred = tf.squeeze(_mag(y_pred), axis=-1)
            log_mag = tf.math.log(mag_pred + 1e-8)
            diff1 = log_mag[..., 1:] - log_mag[..., :-1]

            if use_te_weighting:
                te32 = tf.cast(te, log_mag.dtype)
                dte = te32[1:] - te32[:-1]
                dte = tf.maximum(dte, tf.cast(1e-8, log_mag.dtype))
                slope = diff1 / dte[None, None, None, :]
                dt_mid = 0.5 * (te32[2:] - te32[:-2])
                dt_mid = tf.maximum(dt_mid, tf.cast(1e-8, log_mag.dtype))
                phys_term = (slope[..., 1:] - slope[..., :-1]) / dt_mid[None, None, None, :]
            else:
                phys_term = diff1[..., 1:] - diff1[..., :-1]

            mask = _build_physics_mask(y_true, y_pred)
            if mask is None:
                l_p = tf.reduce_mean(tf.square(phys_term))
            else:
                mask_core = tf.cast(mask[..., 1:-1], phys_term.dtype)
                weighted = tf.square(phys_term) * mask_core
                l_p = tf.reduce_sum(weighted) / (tf.reduce_sum(mask_core) + tf.cast(1e-8, phys_term.dtype))
        else:
            l_p = tf.cast(0.0, y_pred.dtype)

        return lam_c * l_c + lam_m * l_m + lam_p * l_p

    loss.__name__ = f"img_{recon_loss_type}_lc{lam_c:.2f}_lm{lam_m:.2f}_lp{lam_p:.2f}"
    return loss


# ---------------------------------------------------------------------------
# ME-MoC network
# ---------------------------------------------------------------------------

def build_memoc(
    input_shape: Tuple[int, ...],
    output_channel: int = 2,
    filters_root: int = 32,
    conv_times: int = 3,
    up_down_times: int = 4,
    num_echoes: Optional[int] = None,
    attn_heads: int = 2,
    attn_key_dim: int = 16,
    if_relu: bool = False,
    if_residule: bool = False,
) -> Model:
    """Build ME-MoC: complex (2+1)D U-Net with echo-axis attention at the bottleneck.

    Parameters
    ----------
    input_shape   : (H, W, E, 2) -- height, width, echoes, [Re, Im]
    output_channel: must be 2 for complex output (Re, Im)
    filters_root  : complex filters F at the first level (2F real channels)
    conv_times    : factorized blocks per U-Net level
    up_down_times : encoder/decoder depth (H and W must be divisible by 2**depth)
    num_echoes    : E; defaults to input_shape[2]
    attn_heads    : attention heads in EchoAxisAttention
    attn_key_dim  : key dimension per head
    if_relu       : apply CReLU to the final output
    if_residule   : global residual (subtract the network output from the input)
    """
    if output_channel != 2:
        raise ValueError("ME-MoC requires output_channel=2 (complex Re, Im).")
    if len(input_shape) != 4:
        raise ValueError(f"Expected input_shape=(H, W, E, 2), got {input_shape}")
    if num_echoes is None:
        num_echoes = int(input_shape[2])

    def _conv_block(x: tf.Tensor, filters: int, base: str) -> tf.Tensor:
        for i in range(conv_times):
            x = _complex_factorized_block(x, filters, name=f"{base}/FBlock_{i}")
        return x

    def _upconv(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        x = ComplexConv3DTranspose(
            filters, kernel_size=3, padding="same", strides=(2, 2, 1),
            name=f"{name}/CTranspose"
        )(x)
        x = ComplexInstanceNorm(name=f"{name}/UpNorm")(x)
        x = ComplexReLU(name=f"{name}/UpAct")(x)
        return x

    # Input stem: (B,H,W,E,2) -> (B,H,W,E,2*filters_root)
    ipt = Input(input_shape, name="IMGUNet/Input")
    net = ComplexConv3D(filters_root, kernel_size=(3, 3, 1),
                        name="IMGUNet/Stem/SpatialConv")(ipt)
    net = ComplexInstanceNorm(name="IMGUNet/Stem/Norm")(net)
    net = ComplexReLU(name="IMGUNet/Stem/Act")(net)

    # Encoder (pooling halves H and W only; all echoes are preserved).
    skips = []
    for lvl in range(up_down_times):
        filters = (2 ** lvl) * filters_root
        net = _conv_block(net, filters, base=f"IMGUNet/Down_{lvl}")
        skips.append(net)
        net = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1),
                        name=f"IMGUNet/Down_{lvl}/Pool")(net)

    # Bottleneck + echo-axis attention, applied to real and imaginary
    # channels by two separate attention blocks.
    filters = (2 ** up_down_times) * filters_root
    net = _conv_block(net, filters, base="IMGUNet/Bottom")
    net_r, net_i = tf.split(net, num_or_size_splits=2, axis=-1)
    net_r = EchoAxisAttention(
        num_echoes=num_echoes,
        num_heads=attn_heads,
        key_dim=attn_key_dim,
        name="IMGUNet/Bottom/EchoAttn_Re",
    )(net_r)
    net_i = EchoAxisAttention(
        num_echoes=num_echoes,
        num_heads=attn_heads,
        key_dim=attn_key_dim,
        name="IMGUNet/Bottom/EchoAttn_Im",
    )(net_i)
    net = tf.concat([net_r, net_i], axis=-1)

    # Decoder
    for lvl in range(up_down_times - 1, -1, -1):
        filters = (2 ** lvl) * filters_root
        net = _upconv(net, filters, name=f"IMGUNet/Up_{lvl}/UpSample")
        net = Concatenate(axis=-1, name=f"IMGUNet/Up_{lvl}/Skip")([net, skips[lvl]])
        net = _conv_block(net, filters, base=f"IMGUNet/Up_{lvl}")

    # Output head: 2*filters_root complex channels -> 1 complex channel.
    net = ComplexConv3D(1, kernel_size=1, name="IMGUNet/OutConv")(net)
    if if_relu:
        net = ComplexReLU(name="IMGUNet/OutAct")(net)
    if if_residule:
        net = Subtract(name="IMGUNet/Residual")([ipt, net])

    return Model(inputs=ipt, outputs=net, name="ME_MoC")


def build_memoc_from_config(config: dict, input_shape: Tuple[int, ...]) -> Model:
    """Build ME-MoC from a training/inference config (``method.unet3d`` and ``train``)."""
    p = config["method"]["unet3d"]
    train_cfg = config.get("train", {})
    return build_memoc(
        input_shape=input_shape,
        output_channel=int(p.get("output_channel", 2)),
        filters_root=int(p.get("filters_root", 32)),
        conv_times=int(p.get("conv_times", 3)),
        up_down_times=int(p.get("up_down_times", 4)),
        num_echoes=int(input_shape[2]),
        attn_heads=int(train_cfg.get("attn_heads", 2)),
        attn_key_dim=int(train_cfg.get("attn_key_dim", 16)),
        if_relu=bool(p.get("if_relu", False)),
        if_residule=bool(p.get("if_residule", False)),
    )


def compile_memoc(
    model: tf.keras.Model,
    learning_rate: float,
    loss_type: str = "l2",
    lambda_mag: float = 0.2,
    lambda_phys: float = 0.02,
    te_ms: Optional[List[float]] = None,
    use_te_weighting: bool = False,
    physics_mask_mode: str = "echo1_true",
    physics_mask_threshold: float = 0.03,
    psnr_max_val: float = 1.0,
) -> None:
    """Compile ME-MoC with Adam (clipnorm 1.0), the combined loss and magnitude metrics."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate), clipnorm=1.0)
    psnr_max_val = float(psnr_max_val)

    def _mag(t: tf.Tensor) -> tf.Tensor:
        return img_output_magnitude(tf.cast(t, tf.float32))

    def _psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(y_true)
        y_pred_mag = _mag(y_pred)
        if y_true_mag.shape.rank == 3:
            y_true_mag = y_true_mag[..., None]
            y_pred_mag = y_pred_mag[..., None]
        return tf.reduce_mean(tf.image.psnr(y_pred_mag, y_true_mag, max_val=psnr_max_val))

    def _ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_mag = _mag(y_true)
        y_pred_mag = _mag(y_pred)
        if y_true_mag.shape.rank == 3:
            y_true_mag = y_true_mag[..., None]
            y_pred_mag = y_pred_mag[..., None]
        return tf.reduce_mean(tf.image.ssim(y_pred_mag, y_true_mag, max_val=psnr_max_val))

    def _snr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_flat = tf.reshape(_mag(y_true), [-1])
        y_pred_flat = tf.reshape(_mag(y_pred), [-1])
        num = tf.norm(y_true_flat)
        den = tf.norm(y_true_flat - y_pred_flat)
        return 20.0 * (tf.math.log(num / (den + 1e-12)) / tf.math.log(tf.constant(10.0, dtype=y_true_flat.dtype)))

    def _phase(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return img_phase_cosine_error(y_true, y_pred, mask_threshold=float(physics_mask_threshold))

    _psnr.__name__ = "psnr"
    _ssim.__name__ = "ssim"
    _snr.__name__ = "snr"
    _phase.__name__ = "phase_cosine_error"

    loss_fn = make_img_loss(
        lambda_mag=float(lambda_mag),
        lambda_phys=float(lambda_phys),
        loss_type=str(loss_type),
        te_ms=te_ms,
        use_te_weighting=bool(use_te_weighting),
        mask_mode=str(physics_mask_mode),
        mask_threshold=float(physics_mask_threshold),
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[_psnr, _ssim, _snr, _phase])
