import tensorflow as tf
import tf2onnx
import onnx

from neuro_bes.network.FNO import SpectralConv1D, _resolve_dtype, _DTYPE_NAME_MAP


@tf.keras.utils.register_keras_serializable(package="neuro_bes")
class FNO1dMCDropout(tf.keras.Model):
    """
    FNO1d with an always-on dropout before the output projection, for cheap approximate
    Monte-Carlo-dropout uncertainty. Unlike standard dropout, this stays active during both
    training and inference (the two must match for MC dropout's approximate-Bayesian
    interpretation to hold), and it uses `tf.nn.dropout` directly rather than
    `tf.keras.layers.Dropout` -- the Keras layer's underlying stateless RNG op
    (StatelessRandomUniformV2) has no ONNX implementation at all (a hard export failure),
    while `tf.nn.dropout` lowers to RandomUniformLike, a standard op ONNX Runtime supports.

    `call()` on its own returns a single stochastic point estimate of shape (batch,
    data_length), the same shape as FNO1d, so it trains with the same `deepNN.train` /
    `model.fit` flow as a normal FNO. Wrap a trained instance with `MCDropoutUncertaintyWrapper`
    to turn it into an uncertainty estimator for evaluation/deployment.
    """

    def __init__(
        self,
        data_length,
        modes=16,
        width=64,
        fc_units=128,
        dropout_rate=0.1,
        name="FNO1dMCDropout",
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
        self.dropout_rate = dropout_rate
        self.onnx_opset = onnx_opset
        self.output_path = output_path

        self.fc0 = tf.keras.layers.Dense(width, dtype=dtype)

        self.conv0 = SpectralConv1D(width, width, modes, dtype=dtype)
        self.convend = SpectralConv1D(width, width, modes, dtype=dtype)
        self.w0 = tf.keras.layers.Dense(width, dtype=dtype)
        self.w3 = tf.keras.layers.Dense(width, dtype=dtype)

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
        x = tf.nn.dropout(x, rate=self.dropout_rate)
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
            "dropout_rate": self.dropout_rate,
            "onnx_opset": self.onnx_opset,
            "output_path": self.output_path,
            "dtype": _DTYPE_NAME_MAP[self._float_dtype],
        })
        return config


def make_fno_mc_dropout(
    data_length,
    modes=16,
    width=64,
    fc_units=128,
    dropout_rate=0.1,
    name="FNO1dMCDropout",
    onnx_opset=13,
    output_path="model.onnx",
    dtype="float32",
):
    """
    Builds an FNO1d variant with always-on dropout for Monte-Carlo-dropout uncertainty.

    Args:
        data_length (int): Length of the input/output profile (spatial grid resolution).
        modes (int): Number of Fourier modes kept in each spectral layer. Must be <= data_length // 2 + 1.
        width (int): Channel width used throughout the Fourier layers.
        fc_units (int): Number of units in the hidden projection layer before the output.
        dropout_rate (float): Dropout rate applied before the output projection, always active.
        name (str): Model name.
        onnx_opset (int): ONNX opset version for export.
        output_path (str): Default ONNX export path.
        dtype (str): Floating-point precision for all network components -- 'float32' or 'float64'.

    Returns:
        FNO1dMCDropout: The built model. Train it like a normal FNO1d, then wrap it with
        `MCDropoutUncertaintyWrapper` for uncertainty estimation.
    """
    return FNO1dMCDropout(
        data_length=data_length,
        modes=modes,
        width=width,
        fc_units=fc_units,
        dropout_rate=dropout_rate,
        name=name,
        onnx_opset=onnx_opset,
        output_path=output_path,
        dtype=dtype,
    )


@tf.keras.utils.register_keras_serializable(package="neuro_bes")
class MCDropoutUncertaintyWrapper(tf.keras.Model):
    """
    Wraps a trained FNO1dMCDropout to produce cheap approximate uncertainty at inference: draws
    `mc_samples` independent stochastic forward passes by tiling the input into one larger batch
    through the SAME shared weights (only the dropout mask differs per draw, so no separate model
    instances or repeated calls are needed) -- the same "N members for the cost of one batched op"
    trick EnsembleELM_Bayesian uses for its ensemble, applied here to dropout draws instead of
    distinct random models. Reduces the draws to `[mean | std | ci_lower | ci_upper]`, matching
    EnsembleELM_Bayesian's output convention.
    """

    def __init__(
        self,
        base_model,
        mc_samples=20,
        conf_interval_z=1.96,
        name="FNO_MCDropout_wrapper",
        onnx_opset=13,
        output_path="model.onnx",
    ):
        super().__init__(name=name, dtype=base_model.dtype)
        self.base_model = base_model
        self.mc_samples = mc_samples
        self.conf_interval_z = conf_interval_z
        self.onnx_opset = onnx_opset
        self.output_path = output_path
        self.data_length = base_model.data_length
        self._float_dtype = base_model._float_dtype

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=self._float_dtype)
        x_tiled = tf.repeat(x, repeats=self.mc_samples, axis=0)  # (batch * mc_samples, length, 2)
        preds = self.base_model(x_tiled)  # dropout is always active in base_model
        preds = tf.reshape(preds, [-1, self.mc_samples, self.data_length])  # (batch, mc_samples, length)
        mean = tf.reduce_mean(preds, axis=1)
        std = tf.math.reduce_std(preds, axis=1)
        ci_lower = mean - self.conf_interval_z * std
        ci_upper = mean + self.conf_interval_z * std
        return tf.concat([mean, std, ci_lower, ci_upper], axis=1)

    def export_to_onnx(self, output_path=None):
        out = output_path or self.output_path
        input_signature = [tf.TensorSpec([None, self.data_length, 2], self._float_dtype, name="input")]
        onnx_model, _ = tf2onnx.convert.from_keras(self, input_signature=input_signature, opset=self.onnx_opset)
        onnx.save_model(onnx_model, out)
        return out
