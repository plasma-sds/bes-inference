import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

_DTYPE_MAP = {
    "float32": tf.float32,
    "float64": tf.float64,
}
_DTYPE_NAME_MAP = {v: k for k, v in _DTYPE_MAP.items()}


def _resolve_dtype(dtype):
    """Returns (dtype_name, tf_dtype) for a 'float32'/'float64' string or a matching tf.DType.

    The name is forwarded to the Keras base class so its own dtype policy (and therefore its
    automatic input casting) matches the precision requested here, instead of silently
    defaulting to float32.
    """
    if isinstance(dtype, str):
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be 'float32' or 'float64', got '{dtype}'")
        return dtype, _DTYPE_MAP[dtype]
    if dtype in _DTYPE_NAME_MAP:
        return _DTYPE_NAME_MAP[dtype], dtype
    raise ValueError(f"dtype must be 'float32' or 'float64', got '{dtype}'")


@tf.keras.utils.register_keras_serializable(package="neuro_bes")
class SpectralConv1D(tf.keras.layers.Layer):
    """
    1D spectral convolution layer used inside a Fourier Neural Operator.
    Transforms the input to frequency space, applies a learned complex-linear
    transform to the lowest `modes` frequencies, and transforms the (implicitly
    zero-padded) result back to physical space.

    The forward/inverse truncated real-DFT is expressed as a pair of fixed real
    matrices (built once from `numpy.fft`, matching `tf.signal.rfft`/`irfft` bit for
    bit) rather than `tf.signal.rfft`/`irfft` directly, so the whole layer reduces to
    plain matrix multiplications. `tf2onnx` only supports RFFT/IRFFT when the RFFT's
    sole consumer is `ComplexAbs`, which does not hold for a spectral convolution, so
    exporting the FFT ops directly to ONNX fails; matmuls export without issue.

    Operates on channels-last tensors of shape (batch, length, in_channels)
    and returns tensors of shape (batch, length, out_channels).
    """

    def __init__(self, in_channels, out_channels, modes, dtype="float32", **kwargs):
        dtype_name, self._float_dtype = _resolve_dtype(dtype)
        super().__init__(dtype=dtype_name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

    def build(self, input_shape):
        length = int(input_shape[-2])
        max_modes = length // 2 + 1
        if self.modes > max_modes:
            raise ValueError(
                f"modes ({self.modes}) cannot exceed length // 2 + 1 ({max_modes}) for an input length of {length}."
            )
        self._length = length

        np_dtype = np.float32 if self._float_dtype == tf.float32 else np.float64
        num_freqs = length // 2 + 1

        # Forward truncated real-DFT: forward[k, n] = exp(-2*pi*i*k*n/length), k < modes.
        forward = np.fft.rfft(np.eye(length, dtype=np_dtype), axis=-1).T[: self.modes, :]

        # Inverse truncated real-IDFT: the real-space contribution of each retained
        # frequency bin, found via unit impulses so it reproduces tf.signal.irfft's
        # Hermitian-symmetric reconstruction (with the remaining bins implicitly zero)
        # exactly, without ever materializing a complex tensor.
        inverse_real = np.zeros((self.modes, length), dtype=np_dtype)
        inverse_imag = np.zeros((self.modes, length), dtype=np_dtype)
        probe = np.zeros(num_freqs, dtype=np.complex128)
        for k in range(self.modes):
            probe[:] = 0.0
            probe[k] = 1.0
            inverse_real[k] = np.fft.irfft(probe, n=length)
            probe[k] = 1.0j
            inverse_imag[k] = np.fft.irfft(probe, n=length)

        # Stored as non-trainable weights rather than tf.constant: a constant created
        # inside build() is pinned to whatever FuncGraph build() happened to trace
        # under, and becomes inaccessible once call() is traced into a different one
        # (e.g. a model.fit() training step); weights are tracked by the layer itself
        # and stay valid across separate traces.
        self._forward_real = self.add_weight(
            name="forward_real", shape=forward.real.shape,
            initializer=tf.keras.initializers.Constant(forward.real), trainable=False, dtype=self._float_dtype,
        )
        self._forward_imag = self.add_weight(
            name="forward_imag", shape=forward.imag.shape,
            initializer=tf.keras.initializers.Constant(forward.imag), trainable=False, dtype=self._float_dtype,
        )
        self._inverse_real = self.add_weight(
            name="inverse_real", shape=inverse_real.shape,
            initializer=tf.keras.initializers.Constant(inverse_real), trainable=False, dtype=self._float_dtype,
        )
        self._inverse_imag = self.add_weight(
            name="inverse_imag", shape=inverse_imag.shape,
            initializer=tf.keras.initializers.Constant(inverse_imag), trainable=False, dtype=self._float_dtype,
        )

        scale = 1.0 / (self.in_channels * self.out_channels)
        init = tf.random_uniform_initializer(minval=-scale, maxval=scale)
        weight_shape = (self.in_channels, self.out_channels, self.modes)
        self.w_real = self.add_weight(
            name="weights_real", shape=weight_shape, initializer=init, dtype=self._float_dtype, trainable=True
        )
        self.w_imag = self.add_weight(
            name="weights_imag", shape=weight_shape, initializer=init, dtype=self._float_dtype, trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])  # (batch, in_channels, length)

        x_real = tf.einsum("bil,ml->bim", x, self._forward_real)  # (batch, in_channels, modes)
        x_imag = tf.einsum("bil,ml->bim", x, self._forward_imag)

        y_real = tf.einsum("bim,iom->bom", x_real, self.w_real) - tf.einsum("bim,iom->bom", x_imag, self.w_imag)
        y_imag = tf.einsum("bim,iom->bom", x_real, self.w_imag) + tf.einsum("bim,iom->bom", x_imag, self.w_real)

        out = tf.einsum("bom,mn->bon", y_real, self._inverse_real) + tf.einsum(
            "bom,mn->bon", y_imag, self._inverse_imag
        )
        return tf.transpose(out, perm=[0, 2, 1])  # (batch, length, out_channels)

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "modes": self.modes,
            "dtype": _DTYPE_NAME_MAP[self._float_dtype],
        })
        return config


@tf.keras.utils.register_keras_serializable(package="neuro_bes")
class FNO1d(tf.keras.Model):
    """
    1D Fourier Neural Operator (FNO), after Li et al. 2020, mapping one physical-space
    profile onto another of the same spatial grid and resolution (e.g. a plasma density
    profile onto the corresponding beam emission profile).

    Input:  (batch, data_length, 2) — the profile value in channel 0 and a [0, 1]-normalized
            grid-position encoding in channel 1. Use `prepare_fno_input` to build this from a
            `besInferenceDatapoints` object.
    Output: (batch, data_length) — the predicted profile.
    """

    def __init__(
        self,
        data_length,
        modes=16,
        width=64,
        fc_units=128,
        name="FNO1d",
        onnx_opset=13,
        output_path="model.onnx",
        dtype="float32",
        **kwargs,
    ):
        dtype_name, self._float_dtype = _resolve_dtype(dtype)
        super().__init__(name=name, dtype=dtype_name, **kwargs)
        dtype = dtype_name

        max_modes = data_length // 2 + 1
        if modes > max_modes:
            raise ValueError(f"modes ({modes}) cannot exceed data_length // 2 + 1 ({max_modes}).")

        self.data_length = data_length
        self.modes = modes
        self.width = width
        self.fc_units = fc_units
        self.onnx_opset = onnx_opset
        self.output_path = output_path

        self.fc0 = tf.keras.layers.Dense(width, dtype=dtype)

        self.conv0 = SpectralConv1D(width, width, modes, dtype=dtype)
        self.convend = SpectralConv1D(width, width, modes, dtype=dtype)
        self.w0 = tf.keras.layers.Conv1D(width, kernel_size=1, dtype=dtype)
        self.w3 = tf.keras.layers.Conv1D(width, kernel_size=1, dtype=dtype)

        self.fc1 = tf.keras.layers.Dense(fc_units, activation="relu", dtype=dtype)
        self.fc2 = tf.keras.layers.Dense(1, dtype=dtype)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=self._float_dtype)
        x = self.fc0(x)  # (batch, length, width)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = tf.nn.relu(x1 + x2)

        x1 = self.convend(x)
        x2 = self.w3(x)
        x = x1 + x2  # (batch, length, width)

        x = self.fc1(x)
        x = self.fc2(x)
        return tf.squeeze(x, axis=-1)  # (batch, length)

    def export_to_onnx(self, output_path=None):
        out = output_path or self.output_path
        input_signature = [tf.TensorSpec([None, self.data_length, 2], self._float_dtype, name="input")]
        onnx_model, _ = tf2onnx.convert.from_keras(self, input_signature=input_signature, opset=self.onnx_opset)
        onnx.save_model(onnx_model, out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "data_length": self.data_length,
            "modes": self.modes,
            "width": self.width,
            "fc_units": self.fc_units,
            "onnx_opset": self.onnx_opset,
            "output_path": self.output_path,
            "dtype": _DTYPE_NAME_MAP[self._float_dtype],
        })
        return config


def make_fno(
    data_length,
    modes=16,
    width=64,
    fc_units=128,
    name="FNO1d",
    onnx_opset=13,
    output_path="model.onnx",
    dtype="float32",
):
    """
    Builds a 1D Fourier Neural Operator (FNO) model with input/output of the same length.

    Args:
        data_length (int): Length of the input/output profile (spatial grid resolution).
        modes (int): Number of Fourier modes kept in each spectral layer. Must be <= data_length // 2 + 1.
        width (int): Channel width used throughout the Fourier layers.
        fc_units (int): Number of units in the hidden projection layer before the output.
        name (str): Model name.
        onnx_opset (int): ONNX opset version for export.
        output_path (str): Default ONNX export path.
        dtype (str): Floating-point precision for all network components — 'float32' or 'float64'.

    Returns:
        FNO1d: The built FNO model.
    """
    return FNO1d(
        data_length=data_length,
        modes=modes,
        width=width,
        fc_units=fc_units,
        name=name,
        onnx_opset=onnx_opset,
        output_path=output_path,
        dtype=dtype,
    )


def prepare_fno_input(bes_data, channel="density", dtype="float32"):
    """
    Builds the (n_samples, resolution, 2) input tensor expected by FNO1d / make_fno from a
    besInferenceDatapoints object: channel 0 holds the requested profile values and channel 1
    holds the object's spatial grid, normalized to [0, 1], as a position encoding for the
    spectral layers.

    Args:
        bes_data (besInferenceDatapoints): The BES data object holding the profiles and grid.
        channel (str): Which profile to place in channel 0 — 'density' or 'emission'.
        dtype (str): Floating-point precision for the returned array.

    Returns:
        np.ndarray: Array of shape (n_samples, resolution, 2) ready to feed into FNO1d / make_fno.
    """
    if channel == "density":
        profile = bes_data.densities
    elif channel == "emission":
        profile = bes_data.emissions
    else:
        raise ValueError(f"channel must be 'density' or 'emission', got '{channel}'")

    grid = np.asarray(bes_data.grid)
    grid_min, grid_max = grid.min(), grid.max()
    if grid_max == grid_min:
        raise ValueError("bes_data.grid must span a non-zero range to be normalized to [0, 1].")
    coords = (grid - grid_min) / (grid_max - grid_min)

    np_dtype = np.float32 if dtype == "float32" else np.float64
    values = np.asarray(profile).astype(np_dtype)
    coords = np.broadcast_to(coords, values.shape).astype(np_dtype)
    return np.stack([values, coords], axis=-1)
