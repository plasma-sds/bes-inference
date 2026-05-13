import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

tf.keras.mixed_precision.set_global_policy("float32")


class EnsembleELM_Bayesian(tf.keras.Model):
    """
    Custom model implementation of a dual-layer Ensemble Extreme Learning Machine (ELM) with Bayesian uncertainty estimation.
    This model consists of two hidden layers with frozen random weights and a trainable output layer solved via ridge regression.
    The model outputs mean predictions along with uncertainty estimates in the form of confidence intervals.
    """
    def __init__(
        self,
        input_dim,
        hidden_units_1,
        hidden_units_2,
        output_units,
        num_models=50,
        conf_interval_z=1.96,
        alpha=0,
        name="EnsembleELM",
        onnx_opset=13,
        output_path="model.onnx"
    ):
        super().__init__(name=name)
        dtype = tf.float32

        self.input_dim = input_dim
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.output_units = output_units
        self.num_models = num_models
        self.conf_interval_z = conf_interval_z
        self.alpha = alpha
        self.onnx_opset = onnx_opset
        self.output_path = output_path

        # Scaling statistics
        self.input_mean = tf.Variable(tf.zeros((input_dim,), dtype=dtype), trainable=False)
        self.input_std = tf.Variable(tf.ones((input_dim,), dtype=dtype), trainable=False)
        self.output_mean = tf.Variable(tf.zeros((output_units,), dtype=dtype), trainable=False)
        self.output_std = tf.Variable(tf.ones((output_units,), dtype=dtype), trainable=False)

        # First hidden layer — frozen random projection
        np.random.seed(42)
        w1_list, b1_list = [], [] # lists to hold weights and biases for each model in the ensemble
        for _ in range(num_models):
            tf.keras.utils.set_random_seed(np.random.randint(0))
            w_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0 - 1e-7, seed=np.random.randint(0))
            b_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0 - 1e-7, seed=np.random.randint(0))
            w1_list.append(w_init((input_dim, hidden_units_1), dtype=dtype))
            b1_list.append(b_init((hidden_units_1,), dtype=dtype))

        self.w1 = tf.Variable(tf.stack(w1_list), trainable=False)
        self.b1 = tf.Variable(tf.stack(b1_list), trainable=False)
        self.running_mean = tf.Variable(tf.zeros((num_models, hidden_units_1), dtype=dtype), trainable=False)
        self.running_std = tf.Variable(tf.ones((num_models, hidden_units_1), dtype=dtype), trainable=False)

        # Second hidden layer — frozen random projection
        k2_list, b2_list = [], []
        for _ in range(num_models):
            tf.keras.utils.set_random_seed(np.random.randint(0))
            k_init = tf.keras.initializers.RandomUniform(minval=-0.75, maxval=0.75, seed=np.random.randint(0))
            k2_list.append(k_init((hidden_units_1, hidden_units_2), dtype=dtype))
            tf.keras.utils.set_random_seed(np.random.randint(0))
            b_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0 - 1e-7, seed=np.random.randint(0))
            b2_list.append(b_init((hidden_units_2,), dtype=dtype))

        self.k2 = tf.Variable(tf.stack(k2_list), trainable=False)
        self.b2 = tf.Variable(tf.stack(b2_list), trainable=False)

        # Output kernel — solved via ridge regression during training
        self.ko = tf.Variable(tf.zeros((num_models, hidden_units_2, output_units), dtype=dtype), trainable=False)

        # Bayesian posterior
        self.sigma2 = tf.Variable(1.0, trainable=False, dtype=dtype)
        self.post_cov = tf.Variable(tf.zeros((num_models, hidden_units_2, hidden_units_2), dtype=dtype), trainable=False)

    def _scale_inputs(self, x):
        return (x - self.input_mean) / (self.input_std + 1e-8)

    def _scale_outputs(self, y):
        return (y - self.output_mean) / (self.output_std + 1e-8)

    def _unscale_outputs(self, y_scaled):
        return y_scaled * (self.output_std + 1e-8) + self.output_mean

    def build(self, input_shape):
        input_shape = tf.TensorShape([None, self.input_dim])
        super().build(input_shape)
        self._set_inputs(tf.keras.Input(shape=(self.input_dim,), dtype=tf.float32))

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        inputs = self._scale_inputs(inputs)

        s = tf.matmul(tf.expand_dims(inputs, 0), self.w1) + tf.expand_dims(self.b1, 1)
        feature_mean = tf.expand_dims(self.running_mean, axis=1)
        feature_std = tf.expand_dims(self.running_std, axis=1)
        s = (s - feature_mean) / feature_std
        s = s * tf.constant(-0.5, dtype=tf.float32) + tf.constant(-0.5, dtype=tf.float32)
        x = tf.tanh(s)

        h = tf.matmul(x, self.k2) + tf.expand_dims(self.b2, 1)
        h = tf.nn.softmax(h, axis=-1)

        pred = tf.matmul(h, self.ko)
        mean_pred = tf.reduce_mean(pred, axis=0)

        intermediate = tf.matmul(h, self.post_cov)
        var_model = tf.reduce_sum(intermediate * h, axis=-1)
        var_model_mean = tf.reduce_mean(var_model, axis=0)

        total_var_scaled = var_model_mean + self.sigma2
        total_std_scaled = tf.sqrt(total_var_scaled)[..., None]

        mean_pred = tf.nn.relu(self._unscale_outputs(mean_pred))
        std_pred = total_std_scaled * (self.output_std + 1e-8)

        z = self.conf_interval_z  # parameter which is to be calibrated for different confidence intervals
        ci_lower = tf.nn.relu(mean_pred - z * std_pred)
        ci_upper = tf.nn.relu(mean_pred + z * std_pred)

        return tf.concat([mean_pred, std_pred, ci_lower, ci_upper], axis=1)

    def train_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        # fp32 precision workaround for large-magnitude inputs
        scale_factor = 1e18
        x_small = x / scale_factor
        x_mean = tf.reduce_mean(x_small, axis=0)
        x_std = tf.math.reduce_std(x_small, axis=0) + 1e-8

        self.input_mean.assign(x_mean * scale_factor)
        self.input_std.assign(x_std * scale_factor)
        self.output_mean.assign(tf.reduce_mean(y, axis=0))
        self.output_std.assign(tf.math.reduce_std(y, axis=0) + 1e-8)

        x = self._scale_inputs(x)
        y = self._scale_outputs(y)

        s = tf.matmul(tf.expand_dims(x, 0), self.w1) + tf.expand_dims(self.b1, 1)
        feature_mean = tf.reduce_mean(s, axis=1, keepdims=True)
        feature_std = tf.math.reduce_std(s, axis=1, keepdims=True) + 1e-6
        self.running_mean.assign(tf.squeeze(feature_mean, axis=1))
        self.running_std.assign(tf.squeeze(feature_std, axis=1))

        s = (s - feature_mean) / feature_std
        s = s * -0.5 + -0.5
        x = tf.tanh(s)

        H = tf.matmul(x, self.k2) + tf.expand_dims(self.b2, 1)
        H = tf.nn.softmax(H, axis=-1)

        I = tf.expand_dims(tf.eye(self.hidden_units_2, dtype=tf.float32), axis=0)
        HtH = tf.matmul(H, H, transpose_a=True)
        HtY = tf.matmul(H, tf.expand_dims(y, 0), transpose_a=True)

        beta = tf.linalg.solve(HtH + self.alpha * I, HtY)
        self.ko.assign(beta)

        y_pred = tf.matmul(H, beta)
        mean_pred = tf.reduce_mean(y_pred, axis=0)
        sigma2 = tf.reduce_mean(tf.square(y - mean_pred))

        post_cov = tf.linalg.inv(HtH + self.alpha * I) * sigma2
        self.sigma2.assign(sigma2)
        self.post_cov.assign(post_cov)

        return {"loss": tf.reduce_mean(tf.square(y - mean_pred))}

    def export_to_onnx(self, output_path=None):
        out = output_path or self.output_path
        onnx_model, _ = tf2onnx.convert.from_keras(self, opset=self.onnx_opset)
        onnx.save_model(onnx_model, out)
        return out


def make_ensemble_elm(
    data_length,
    units_per_layer=[200, 200],
    num_models=50,
    conf_interval_z=1.96,
    alpha=0,
    name="VectorizedEnsembleELM",
    onnx_opset=13,
    output_path="model.onnx"
):
    """
    Builds an ensemble ELM with Bayesian uncertainty estimation.

    Args:
        data_length (int): Length of the input/output data.
        units_per_layer (list of 2 ints): Neurons in each hidden layer [hidden_1, hidden_2].
        num_models (int): Number of ELM models in the ensemble.
        conf_interval_z (float): Z-score for the desired confidence interval. To be calibrated for the specific problem, assuming normal distribution of errors.
        alpha (float): Ridge regression regularization coefficient.
        name (str): Model name.
        onnx_opset (int): ONNX opset version for export.
        output_path (str): Default ONNX export path.

    Returns:
        EnsembleELM_Bayesian: The built ensemble ELM model.
    """
    if len(units_per_layer) != 2:
        raise ValueError("units_per_layer must have exactly 2 elements [hidden_1, hidden_2].")

    return EnsembleELM_Bayesian(
        input_dim=data_length,
        hidden_units_1=units_per_layer[0],
        hidden_units_2=units_per_layer[1],
        output_units=data_length,
        num_models=num_models,
        conf_interval_z=conf_interval_z,
        alpha=alpha,
        name=name,
        onnx_opset=onnx_opset,
        output_path=output_path
    )


def train(model, x, y):
    """
    Trains the ensemble ELM on the full dataset in a single closed-form pass.

    Unlike gradient-based models, ELM solves for output weights analytically via
    ridge regression, so the entire dataset must be passed as a single batch.

    Args:
        model (EnsembleELM_Bayesian): The ELM model to train.
        x (np.ndarray): Input data of shape (n_samples, data_length).
        y (np.ndarray): Target data of shape (n_samples, data_length).

    Returns:
        EnsembleELM_Bayesian: The trained model.
    """
    model.compile()
    model.fit(x, y, batch_size=len(x), epochs=1)
    return model
