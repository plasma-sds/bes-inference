import tensorflow as tf
import tf2onnx
import onnx

_DTYPE_MAP = {
    "float32": tf.float32,
    "float64": tf.float64,
}
_DTYPE_NAME_MAP = {v: k for k, v in _DTYPE_MAP.items()}


def _resolve_dtype(dtype):
    if isinstance(dtype, str):
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be 'float32' or 'float64', got '{dtype}'")
        return dtype, _DTYPE_MAP[dtype]
    if dtype in _DTYPE_NAME_MAP:
        return _DTYPE_NAME_MAP[dtype], dtype
    raise ValueError(f"dtype must be 'float32' or 'float64', got '{dtype}'")


@tf.keras.utils.register_keras_serializable(package="neuro_bes")
class EnsembleMLP(tf.keras.Model):
    """
    An ensemble of `num_models` independently-initialized deep MLPs (the "less wide, more deep"
    architecture Johannes's email suggested as a replacement for ELM's wide/shallow design),
    trained together and combined for a more accurate mean prediction plus an empirical
    uncertainty estimate -- the standard "deep ensembles" approach (Lakshminarayanan et al.),
    applied with `EnsembleELM_Bayesian`'s own implementation trick: every member's weights carry a
    leading `num_models` dimension and every layer is one batched matmul across all members at
    once (see `neuro_bes/network/elm.py`), rather than looping over `num_models` separate model
    instances -- the difference between the ensemble costing roughly the same *ratio* ELM's does
    (~4x a single small model, measured in `fno_uncertainty_benchmark.ipynb`) versus literally
    `num_models` times as much (measured directly for a naive sequential ensemble earlier in this
    project: cost scales almost exactly linearly with member count).

    Unlike `EnsembleELM_Bayesian` (frozen random hidden layers, closed-form ridge-regression output
    layer) or `FNO1dBayesianHead` (closed-form head on frozen backprop-trained features), every
    layer of every member here is trained jointly via ordinary backpropagation -- diversity comes
    entirely from each member's independent random initialization, exactly as in standard deep
    ensembles. The training loss is the mean squared error of each member against the target,
    averaged over members (not the loss of the averaged prediction), so each member is pushed to
    independently fit the data rather than just contribute to a shared point estimate.

    `call()` returns `[mean | std | ci_lower | ci_upper]`, each `data_length` wide, matching the
    output convention of `EnsembleELM_Bayesian`, `FNO1dMCDropout`/`MCDropoutUncertaintyWrapper`,
    and `FNO1dBayesianHead` -- the standard deviation here is the empirical spread across members,
    not an analytic Bayesian posterior.
    """

    def __init__(
        self,
        data_length,
        num_layers=4,
        units_per_layer=32,
        activation="relu",
        output_activation=None,
        num_models=20,
        conf_interval_z=1.96,
        name="EnsembleMLP",
        onnx_opset=13,
        output_path="model.onnx",
        dtype="float32",
        **kwargs,
    ):
        dtype_name, self._float_dtype = _resolve_dtype(dtype)
        super().__init__(name=name, dtype=dtype_name, **kwargs)

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.data_length = data_length
        self.num_layers = num_layers
        self.units_per_layer = units_per_layer
        self.activation = tf.keras.activations.get(activation)
        self.output_activation = tf.keras.activations.get(output_activation)
        self.num_models = num_models
        self.conf_interval_z = conf_interval_z
        self.onnx_opset = onnx_opset
        self.output_path = output_path

        layer_dims = [data_length] + [units_per_layer] * (num_layers - 1) + [data_length]
        self._weights = []
        self._biases = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            w_init = tf.keras.initializers.GlorotUniform()
            b_init = tf.keras.initializers.Zeros()
            w = self.add_weight(
                name=f"w{i}", shape=(num_models, in_dim, out_dim), initializer=w_init, dtype=self._float_dtype
            )
            b = self.add_weight(
                name=f"b{i}", shape=(num_models, out_dim), initializer=b_init, dtype=self._float_dtype
            )
            self._weights.append(w)
            self._biases.append(b)

    def _raw_predictions(self, x):
        """Returns every member's own prediction, shape (num_models, batch, data_length)."""
        x = tf.convert_to_tensor(x, dtype=self._float_dtype)
        h = tf.expand_dims(x, 0)  # (1, batch, data_length), broadcasts against (num_models, ...) below
        n_layers = len(self._weights)
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            h = tf.matmul(h, w) + tf.expand_dims(b, 1)  # (num_models, batch, out_dim)
            if i < n_layers - 1:
                h = self.activation(h)
            elif self.output_activation is not None:
                h = self.output_activation(h)
        return h  # (num_models, batch, data_length)

    def call(self, inputs):
        raw = self._raw_predictions(inputs)  # (num_models, batch, data_length)
        mean_pred = tf.reduce_mean(raw, axis=0)
        std_pred = tf.math.reduce_std(raw, axis=0)
        ci_lower = mean_pred - self.conf_interval_z * std_pred
        ci_upper = mean_pred + self.conf_interval_z * std_pred
        return tf.concat([mean_pred, std_pred, ci_lower, ci_upper], axis=1)

    def train_step(self, data):
        x, y = data
        x = tf.cast(x, self._float_dtype)
        y = tf.cast(y, self._float_dtype)
        with tf.GradientTape() as tape:
            raw = self._raw_predictions(x)  # (num_models, batch, data_length)
            loss = tf.reduce_mean(tf.square(raw - tf.expand_dims(y, 0)))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        x = tf.cast(x, self._float_dtype)
        y = tf.cast(y, self._float_dtype)
        raw = self._raw_predictions(x)
        loss = tf.reduce_mean(tf.square(raw - tf.expand_dims(y, 0)))
        return {"loss": loss}

    def export_to_onnx(self, output_path=None):
        out = output_path or self.output_path
        input_signature = [tf.TensorSpec([None, self.data_length], self._float_dtype, name="input")]
        onnx_model, _ = tf2onnx.convert.from_keras(self, input_signature=input_signature, opset=self.onnx_opset)
        onnx.save_model(onnx_model, out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "data_length": self.data_length,
            "num_layers": self.num_layers,
            "units_per_layer": self.units_per_layer,
            "activation": tf.keras.activations.serialize(self.activation),
            "output_activation": tf.keras.activations.serialize(self.output_activation) if self.output_activation else None,
            "num_models": self.num_models,
            "conf_interval_z": self.conf_interval_z,
            "onnx_opset": self.onnx_opset,
            "output_path": self.output_path,
            "dtype": _DTYPE_NAME_MAP[self._float_dtype],
        })
        return config


def make_ensemble_mlp(
    data_length,
    num_layers=4,
    units_per_layer=32,
    activation="relu",
    output_activation=None,
    num_models=20,
    conf_interval_z=1.96,
    name="EnsembleMLP",
    onnx_opset=13,
    output_path="model.onnx",
    dtype="float32",
):
    """
    Builds an ensemble of `num_models` independently-initialized deep MLPs, trained jointly via
    backpropagation, for a more accurate mean prediction than a single MLP plus an empirical
    uncertainty band -- see `EnsembleMLP` for the batched-ensemble implementation.

    Args:
        data_length (int): Length of the input/output profile.
        num_layers (int): Number of layers per ensemble member (matching `deepNN.make_MLP`'s
            `num_layers`, i.e. hidden layers before the final output projection).
        units_per_layer (int): Neurons per hidden layer, shared by every member.
        activation (str): Hidden-layer activation.
        output_activation (str or None): Output-layer activation.
        num_models (int): Number of ensemble members.
        conf_interval_z (float): Z-score for the reported confidence interval.
        name (str): Model name.
        onnx_opset (int): ONNX opset version for export.
        output_path (str): Default ONNX export path.
        dtype (str): Floating-point precision -- 'float32' or 'float64'.

    Returns:
        EnsembleMLP: The built model. Train it like any other Keras model, e.g. with
        `neuro_bes.network.deepNN.train`.
    """
    return EnsembleMLP(
        data_length=data_length,
        num_layers=num_layers,
        units_per_layer=units_per_layer,
        activation=activation,
        output_activation=output_activation,
        num_models=num_models,
        conf_interval_z=conf_interval_z,
        name=name,
        onnx_opset=onnx_opset,
        output_path=output_path,
        dtype=dtype,
    )
