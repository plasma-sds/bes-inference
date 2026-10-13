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
class BIPMLP(tf.keras.Model):
    """
    Input passes through one frozen random projection whose pre-activations are calibrated by
    batch intrinsic plasticity (BIP) -- the exact same mechanism EnsembleELM_Bayesian uses for its
    own first hidden layer (see neuro_bes/network/elm.py's `w1`/`b1`/`running_mean`/`running_std`
    and the `(s - mean) / std * -0.5 + -0.5` affine feeding `tanh`) -- followed by several ordinary
    trainable Dense layers, matching `deepNN.make_MLP`'s hidden-layer stack.

    Unlike EnsembleELM_Bayesian: there's no ensembling (`num_models`), no second frozen random
    layer, no closed-form ridge output, and no internal raw-magnitude input/output rescaling --
    this model expects already-scaled input/output (e.g. via `GlobalMaxScaler`, the convention
    `deepNN.make_MLP`/`EnsembleMLP`/`MixedEnsemble`'s MLP members use), not ELM's raw
    ~1e19-magnitude density directly. Porting that rescaling over is a small addition if you want
    this model to also accept raw data directly -- flagging it as a deliberate scope cut, not an
    oversight.

    Training is two phases, matching the fit-once-then-train-normally split:
      1. `fit_bip(X)`: runs the *entire* training set through the frozen random projection once
         and sets `running_mean`/`running_std` from those per-neuron statistics -- unlike
         EnsembleELM_Bayesian, which recomputes these every `train_step` (in practice ELM is
         always trained in one giant batch anyway, so this is the same computation, just made an
         explicit one-shot call instead of something baked into the training loop).
      2. Ordinary `model.compile(...)` + `model.fit(X, Y, batch_size=..., epochs=...)`: `w1`, `b1`,
         `running_mean`, and `running_std` are all non-trainable weights, so Keras's default
         gradient tape only ever updates the Dense head's own weights -- the BIP front end stays
         exactly as `fit_bip` left it no matter how many mini-batches or epochs the head trains for.
    """

    def __init__(
        self,
        data_length,
        hidden_units_bip=200,
        num_layers=4,
        units_per_layer=32,
        activation="relu",
        output_activation=None,
        name="BIPMLP",
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
        self.hidden_units_bip = hidden_units_bip
        self.num_layers = num_layers
        self.units_per_layer = units_per_layer
        self.activation = activation
        self.output_activation = output_activation
        self.onnx_opset = onnx_opset
        self.output_path = output_path
        self.bip_fitted = False

        # Frozen random projection + BIP normalization -- identical mechanism to EnsembleELM_Bayesian,
        # minus the ensembling dimension (this is a single network, not an ensemble of them).
        w_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0 - 1e-7)
        b_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0 - 1e-7)
        self.w1 = self.add_weight(
            name="w1", shape=(data_length, hidden_units_bip), initializer=w_init,
            trainable=False, dtype=self._float_dtype,
        )
        self.b1 = self.add_weight(
            name="b1", shape=(hidden_units_bip,), initializer=b_init,
            trainable=False, dtype=self._float_dtype,
        )
        self.running_mean = self.add_weight(
            name="running_mean", shape=(hidden_units_bip,), initializer="zeros",
            trainable=False, dtype=self._float_dtype,
        )
        self.running_std = self.add_weight(
            name="running_std", shape=(hidden_units_bip,), initializer="ones",
            trainable=False, dtype=self._float_dtype,
        )

        # Trainable MLP head, matching deepNN.make_MLP's layer-stacking convention.
        activation_fn = tf.keras.activations.get(activation)
        output_activation_fn = tf.keras.activations.get(output_activation)
        layer_dims = [hidden_units_bip] + [units_per_layer] * (num_layers - 1) + [data_length]
        self.head_layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            act = activation_fn if i < num_layers - 1 else output_activation_fn
            self.head_layers.append(tf.keras.layers.Dense(out_dim, activation=act, dtype=self._float_dtype))

    def _bip_features(self, x):
        x = tf.convert_to_tensor(x, dtype=self._float_dtype)
        s = tf.matmul(x, self.w1) + self.b1
        s = (s - self.running_mean) / self.running_std
        s = s * tf.constant(-0.5, dtype=self._float_dtype) + tf.constant(-0.5, dtype=self._float_dtype)
        return tf.tanh(s)

    def fit_bip(self, X):
        """
        Fits `running_mean`/`running_std` from the *entire* array `X` in one shot (not a
        mini-batch), matching EnsembleELM_Bayesian's own BIP statistics. Call this once, before
        `model.compile(...)`/`model.fit(...)`, which train only the Dense head.
        """
        X = tf.convert_to_tensor(X, dtype=self._float_dtype)
        s = tf.matmul(X, self.w1) + self.b1
        feature_mean = tf.reduce_mean(s, axis=0)
        feature_std = tf.math.reduce_std(s, axis=0) + 1e-6
        self.running_mean.assign(feature_mean)
        self.running_std.assign(feature_std)
        self.bip_fitted = True

    def call(self, inputs):
        h = self._bip_features(inputs)
        for layer in self.head_layers:
            h = layer(h)
        return h

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
            "hidden_units_bip": self.hidden_units_bip,
            "num_layers": self.num_layers,
            "units_per_layer": self.units_per_layer,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "onnx_opset": self.onnx_opset,
            "output_path": self.output_path,
            "dtype": _DTYPE_NAME_MAP[self._float_dtype],
        })
        return config


def make_bip_mlp(
    data_length,
    hidden_units_bip=200,
    num_layers=4,
    units_per_layer=32,
    activation="relu",
    output_activation=None,
    name="BIPMLP",
    onnx_opset=13,
    output_path="model.onnx",
    dtype="float32",
):
    """
    Builds a BIPMLP: a frozen random projection calibrated by batch intrinsic plasticity (BIP,
    the same mechanism EnsembleELM_Bayesian's own first hidden layer uses) feeding a stack of
    ordinary trainable Dense layers.

    Call `model.fit_bip(X_train)` once on the *entire* training set before compiling/training --
    this sets the BIP normalization statistics and leaves them fixed. Then use it like any other
    Keras model: `model.compile(optimizer=..., loss="mse")` and
    `model.fit(X_train, Y_train, batch_size=..., epochs=...)` train only the Dense head, since the
    BIP front end's weights are all non-trainable.

    Args:
        data_length (int): Length of the input/output profile.
        hidden_units_bip (int): Width of the frozen random projection / BIP layer.
        num_layers (int): Number of Dense layers in the trainable head (hidden layers plus the
            final output projection, matching deepNN.make_MLP's `num_layers` convention).
        units_per_layer (int): Neurons per hidden Dense layer.
        activation (str): Hidden-layer activation.
        output_activation (str or None): Output-layer activation.
        name (str): Model name.
        onnx_opset (int): ONNX opset version for export.
        output_path (str): Default ONNX export path.
        dtype (str): Floating-point precision -- 'float32' or 'float64'.

    Returns:
        BIPMLP: The built model, with an untrained (zero-initialized) BIP front end -- call
        `fit_bip(X_train)` before training.
    """
    return BIPMLP(
        data_length=data_length,
        hidden_units_bip=hidden_units_bip,
        num_layers=num_layers,
        units_per_layer=units_per_layer,
        activation=activation,
        output_activation=output_activation,
        name=name,
        onnx_opset=onnx_opset,
        output_path=output_path,
        dtype=dtype,
    )
