import tensorflow as tf
import tf2onnx
import onnx


@tf.keras.utils.register_keras_serializable(package="neuro_bes")
class MLPBayesianHead(tf.keras.Model):
    """
    A deepNN.make_MLP network whose final Dense output layer is replaced with a closed-form,
    ridge-regression-solved linear head with an analytic Bayesian posterior -- the same math
    EnsembleELM_Bayesian uses for its output layer (see neuro_bes.network.elm), applied once (no
    ensembling) to the MLP's own learned hidden features instead of ELM's random projections.
    Gives `[mean | std | ci_lower | ci_upper]` at inference for the cost of one extra small matmul
    beyond the MLP's own forward pass -- the fastest of the uncertainty options tried in this
    project (see `fno_uncertainty_benchmark.ipynb`: the equivalent FNO Bayesian head cost only
    ~9% over a plain FNO, versus ~4x for MC-dropout or the ELM ensemble).

    Like EnsembleELM_Bayesian, the predictive variance is a single scalar per profile (from the
    quadratic form h @ post_cov @ h^T), broadcast across all output positions, not an
    independently-varying-per-position estimate -- this is what keeps the closed-form solve to a
    small `feature_dim x feature_dim` matrix regardless of profile length.

    Training is two phases:
      1. Build and train a plain Sequential MLP (via `deepNN.make_MLP` + `deepNN.train`); its own
         final Dense output layer is only a scaffold used to shape the hidden features and is
         discarded afterwards.
      2. Wrap it with `MLPBayesianHead(trunk)` and call `fit_head(X, Y)`, which freezes the hidden
         layers, extracts their output over the whole training set, and solves the output layer
         in closed form.
    """

    def __init__(
        self,
        trunk,
        conf_interval_z=1.96,
        alpha=1e-6,
        name="MLPBayesianHead",
        onnx_opset=13,
        output_path="model.onnx",
    ):
        super().__init__(name=name, dtype=trunk.dtype)
        self.hidden_layers = trunk.layers[:-1]  # everything but the trunk's own output Dense layer
        for layer in self.hidden_layers:
            layer.trainable = False

        self.data_length = trunk.layers[-1].units
        self.feature_dim = self.hidden_layers[-1].units
        self._float_dtype = tf.as_dtype(trunk.dtype)
        self.conf_interval_z = conf_interval_z
        self.alpha = alpha
        self.onnx_opset = onnx_opset
        self.output_path = output_path

        self.ko = self.add_weight(
            name="ko",
            shape=(self.feature_dim, self.data_length),
            initializer="zeros",
            trainable=False,
            dtype=self._float_dtype,
        )
        self.sigma2 = tf.Variable(1.0, trainable=False, dtype=self._float_dtype, name="sigma2")
        self.post_cov = self.add_weight(
            name="post_cov",
            shape=(self.feature_dim, self.feature_dim),
            initializer="zeros",
            trainable=False,
            dtype=self._float_dtype,
        )

    def _features(self, x):
        h = tf.convert_to_tensor(x, dtype=self._float_dtype)
        for layer in self.hidden_layers:
            h = layer(h)
        return h  # (batch, feature_dim)

    def fit_head(self, X, Y):
        H = self._features(X)  # (batch, feature_dim)
        Y = tf.cast(Y, self._float_dtype)  # (batch, data_length)

        eye = tf.eye(self.feature_dim, dtype=self._float_dtype)
        HtH = tf.matmul(H, H, transpose_a=True)  # (feature_dim, feature_dim)
        HtY = tf.matmul(H, Y, transpose_a=True)  # (feature_dim, data_length)
        regularized = HtH + self.alpha * eye

        beta = tf.linalg.solve(regularized, HtY)  # (feature_dim, data_length)
        self.ko.assign(beta)

        y_pred = tf.matmul(H, beta)
        sigma2 = tf.reduce_mean(tf.square(Y - y_pred))
        post_cov = tf.linalg.inv(regularized) * sigma2
        self.sigma2.assign(sigma2)
        self.post_cov.assign(post_cov)

    def call(self, inputs):
        H = self._features(inputs)  # (batch, feature_dim)
        mean_pred = tf.matmul(H, self.ko)  # (batch, data_length)

        intermediate = tf.matmul(H, self.post_cov)  # (batch, feature_dim)
        var_model = tf.reduce_sum(intermediate * H, axis=-1, keepdims=True)  # (batch, 1)
        std_pred = tf.sqrt(var_model + self.sigma2)
        std_pred = tf.broadcast_to(std_pred, tf.shape(mean_pred))

        ci_lower = mean_pred - self.conf_interval_z * std_pred
        ci_upper = mean_pred + self.conf_interval_z * std_pred
        return tf.concat([mean_pred, std_pred, ci_lower, ci_upper], axis=1)

    def export_to_onnx(self, output_path=None):
        out = output_path or self.output_path
        # make_MLP builds input/output of the same length, so data_length doubles as the input dim.
        input_signature = [tf.TensorSpec([None, self.data_length], self._float_dtype, name="input")]
        onnx_model, _ = tf2onnx.convert.from_keras(self, input_signature=input_signature, opset=self.onnx_opset)
        onnx.save_model(onnx_model, out)
        return out


def make_mlp_bayesian_head(trunk, conf_interval_z=1.96, alpha=1e-6, name="MLPBayesianHead", onnx_opset=13, output_path="model.onnx"):
    """
    Wraps a trained deepNN.make_MLP model with a closed-form Bayesian ridge-regression head.
    Call `fit_head(X, Y)` on the result before using it for predictions.

    Args:
        trunk (tf.keras.Sequential): A trained make_MLP model; its own final Dense output layer
            is discarded and its hidden layers are reused as a frozen feature extractor.
        conf_interval_z (float): Z-score for the desired confidence interval.
        alpha (float): Ridge regression regularization coefficient.
        name (str): Model name.
        onnx_opset (int): ONNX opset version for export.
        output_path (str): Default ONNX export path.

    Returns:
        MLPBayesianHead: The wrapped model, not yet fit -- call `fit_head(X, Y)` next.
    """
    return MLPBayesianHead(
        trunk=trunk,
        conf_interval_z=conf_interval_z,
        alpha=alpha,
        name=name,
        onnx_opset=onnx_opset,
        output_path=output_path,
    )
