import tensorflow as tf
import tf2onnx
import onnx


def _fno_trunk_features(trunk, x):
    """
    Runs a trained FNO1d's feature-extracting layers (everything up to, but not including,
    its final Dense(1) output projection), returning per-position features of shape
    (batch, length, fc_units).
    """
    x = tf.convert_to_tensor(x, dtype=trunk._float_dtype)
    x = trunk.fc0(x)

    x1 = trunk.conv0(x)
    x2 = trunk.w0(x)
    x = tf.nn.relu(x1 + x2)

    x1 = trunk.convend(x)
    x2 = trunk.w3(x)
    x = x1 + x2

    return trunk.fc1(x)  # (batch, length, fc_units)


@tf.keras.utils.register_keras_serializable(package="neuro_bes")
class FNO1dBayesianHead(tf.keras.Model):
    """
    An FNO1d whose final projection is replaced with a closed-form, ridge-regression-solved
    linear head with an analytic Bayesian posterior -- the same math EnsembleELM_Bayesian uses
    for its output layer (see `neuro_bes.network.elm`), applied once (no ensembling) to FNO's own
    learned per-position features instead of ELM's random projections. Gives
    `[mean | std | ci_lower | ci_upper]` at inference for the cost of one extra small matmul
    beyond a normal FNO forward pass -- no repeated sampling or ensemble members needed.

    The regression problem pools every (profile, grid-position) pair in the training set as one
    training example, so the head stays a single small `fc_units x fc_units` linear map shared
    across grid positions -- matching FNO's own translation-equivariant weight-sharing structure,
    and keeping the ridge-regression matrix to invert small regardless of profile length.

    Training is two phases:
      1. Build and train a plain `FNO1d` (via `make_fno` + `deepNN.train`, its own Dense(1) head
         is only a scaffold used to shape the features and is discarded afterwards).
      2. Wrap it with `FNO1dBayesianHead(trunk)` and call `fit_head(X, Y)`, which freezes the
         trunk, extracts its features over the whole training set, and solves the output layer in
         closed form.
    """

    def __init__(
        self,
        trunk,
        conf_interval_z=1.96,
        alpha=1e-6,
        name="FNO1dBayesianHead",
        onnx_opset=13,
        output_path="model.onnx",
    ):
        super().__init__(name=name, dtype=trunk.dtype)
        self.trunk = trunk
        self.trunk.trainable = False
        self.data_length = trunk.data_length
        self.fc_units = trunk.fc_units
        self._float_dtype = trunk._float_dtype
        self.conf_interval_z = conf_interval_z
        self.alpha = alpha
        self.onnx_opset = onnx_opset
        self.output_path = output_path

        self.ko = self.add_weight(
            name="ko", shape=(self.fc_units, 1), initializer="zeros", trainable=False, dtype=self._float_dtype
        )
        self.sigma2 = tf.Variable(1.0, trainable=False, dtype=self._float_dtype, name="sigma2")
        self.post_cov = self.add_weight(
            name="post_cov",
            shape=(self.fc_units, self.fc_units),
            initializer="zeros",
            trainable=False,
            dtype=self._float_dtype,
        )

    def fit_head(self, X, Y):
        """
        Solves the output layer in closed form via ridge regression over the trunk's
        per-position features, pooling every (profile, grid-position) pair as one training
        example. Also computes the analytic Bayesian posterior (`sigma2`, `post_cov`) used for
        the std/CI outputs at inference, mirroring `EnsembleELM_Bayesian.train_step`.
        """
        H = _fno_trunk_features(self.trunk, X)  # (batch, length, fc_units)
        H_flat = tf.reshape(H, [-1, self.fc_units])  # (batch * length, fc_units)
        Y_flat = tf.reshape(tf.cast(Y, self._float_dtype), [-1, 1])  # (batch * length, 1)

        eye = tf.eye(self.fc_units, dtype=self._float_dtype)
        HtH = tf.matmul(H_flat, H_flat, transpose_a=True)
        HtY = tf.matmul(H_flat, Y_flat, transpose_a=True)
        regularized = HtH + self.alpha * eye

        beta = tf.linalg.solve(regularized, HtY)
        self.ko.assign(beta)

        y_pred = tf.matmul(H_flat, beta)
        sigma2 = tf.reduce_mean(tf.square(Y_flat - y_pred))
        post_cov = tf.linalg.inv(regularized) * sigma2
        self.sigma2.assign(sigma2)
        self.post_cov.assign(post_cov)

    def call(self, inputs):
        H = _fno_trunk_features(self.trunk, inputs)  # (batch, length, fc_units)

        mean_pred = tf.squeeze(tf.matmul(H, self.ko), axis=-1)  # (batch, length)

        intermediate = tf.matmul(H, self.post_cov)  # (batch, length, fc_units)
        var_model = tf.reduce_sum(intermediate * H, axis=-1)  # (batch, length)
        std_pred = tf.sqrt(var_model + self.sigma2)

        ci_lower = mean_pred - self.conf_interval_z * std_pred
        ci_upper = mean_pred + self.conf_interval_z * std_pred
        return tf.concat([mean_pred, std_pred, ci_lower, ci_upper], axis=1)

    def export_to_onnx(self, output_path=None):
        out = output_path or self.output_path
        input_signature = [tf.TensorSpec([None, self.data_length, 2], self._float_dtype, name="input")]
        onnx_model, _ = tf2onnx.convert.from_keras(self, input_signature=input_signature, opset=self.onnx_opset)
        onnx.save_model(onnx_model, out)
        return out


def make_fno_bayesian_head(trunk, conf_interval_z=1.96, alpha=1e-6, name="FNO1dBayesianHead", onnx_opset=13, output_path="model.onnx"):
    """
    Wraps a trained FNO1d (see `neuro_bes.network.FNO.make_fno`) with a closed-form Bayesian
    ridge-regression head. Call `fit_head(X, Y)` on the result before using it for predictions.

    Args:
        trunk (FNO1d): A trained FNO1d whose feature-extracting layers (up to fc1) are reused;
            its own output Dense(1) layer is discarded.
        conf_interval_z (float): Z-score for the desired confidence interval.
        alpha (float): Ridge regression regularization coefficient.
        name (str): Model name.
        onnx_opset (int): ONNX opset version for export.
        output_path (str): Default ONNX export path.

    Returns:
        FNO1dBayesianHead: The wrapped model, not yet fit -- call `fit_head(X, Y)` next.
    """
    return FNO1dBayesianHead(
        trunk=trunk,
        conf_interval_z=conf_interval_z,
        alpha=alpha,
        name=name,
        onnx_opset=onnx_opset,
        output_path=output_path,
    )
