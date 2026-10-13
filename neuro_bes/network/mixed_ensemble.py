import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

from neuro_bes.network.MLP_ensemble import make_ensemble_mlp
from neuro_bes.network.elm import make_ensemble_elm
from neuro_bes.network.FNO import FNO1d, _resolve_dtype, _DTYPE_NAME_MAP


@tf.keras.utils.register_keras_serializable(package="neuro_bes")
class MixedEnsemble(tf.keras.Model):
    """
    A heterogeneous ensemble of `num_mlp` deep MLPs (`EnsembleMLP`), `num_elm` ELM members
    (`EnsembleELM_Bayesian`), and `num_fno` FNOs (`FNO1d`), predicting `[mean | std | ci_lower |
    ci_upper]` combined across whichever families are included -- different architectures tend to
    make different kinds of errors, so this can improve on a same-architecture ensemble's mean
    accuracy, at the cost of combining multiple training procedures and input conventions into one
    model. Any one or two of `num_mlp`/`num_elm`/`num_fno` may be 0 to leave that family out
    entirely (e.g. `num_fno=0` for an MLP+ELM-only ensemble); at least one must be positive.

    How much each family contributes to the combined mean/std is controlled by `family_weights`
    (see `set_family_weights`), not just member counts: by default (`family_weights=None`) every
    individual member is weighted equally, so a family's influence is simply proportional to how
    many members it has. Pass an explicit `family_weights` dict to instead give each family a
    fixed share of the combined prediction regardless of its member count -- this can be changed
    on an already-trained model via `set_family_weights` without retraining, since combination
    weighting doesn't affect what each sub-model already learned.

    Takes one shared *raw* density input `(batch, data_length)`, matching `EnsembleELM_Bayesian`'s
    own convention (it standardizes internally; no external preprocessing needed for it). The MLP
    and FNO members expect `[0, 1]`-scaled density instead and predict *scaled* emission, so
    `fit(X, Y)` fits simple global-max input/output scales (as
    `neuro_bes.preprocessing.profile_transform.GlobalMaxScaler` does) once, stores them, and
    `call()` translates automatically -- callers only ever see raw units in and out.

    Cost: the ELM and MLP portions are each internally vectorized (one batched matmul across every
    member of that type, the trick `EnsembleELM_Bayesian`/`EnsembleMLP` already use), but no
    batched-ensemble FNO exists yet, so the `num_fno` members are separate `FNO1d` instances and
    that portion's cost scales roughly linearly with `num_fno`. Keep `num_fno` modest if inference
    latency matters.
    """

    def __init__(
        self,
        data_length,
        num_mlp=10,
        num_elm=10,
        num_fno=5,
        mlp_num_layers=4,
        mlp_units_per_layer=32,
        elm_units_per_layer=(200, 200),
        fno_modes=10,
        fno_width=32,
        fno_fc_units=64,
        conf_interval_z=1.96,
        family_weights=None,
        name="MixedEnsemble",
        onnx_opset=13,
        output_path="model.onnx",
        dtype="float32",
        **kwargs,
    ):
        dtype_name, self._float_dtype = _resolve_dtype(dtype)
        super().__init__(name=name, dtype=dtype_name, **kwargs)
        dtype = dtype_name

        if num_mlp < 0 or num_elm < 0 or num_fno < 0:
            raise ValueError("num_mlp, num_elm, and num_fno must each be >= 0.")
        if num_mlp + num_elm + num_fno < 1:
            raise ValueError("at least one of num_mlp, num_elm, num_fno must be >= 1.")

        self.data_length = data_length
        self.num_mlp = num_mlp
        self.num_elm = num_elm
        self.num_fno = num_fno
        self.mlp_num_layers = mlp_num_layers
        self.mlp_units_per_layer = mlp_units_per_layer
        self.elm_units_per_layer = list(elm_units_per_layer)
        self.fno_modes = fno_modes
        self.fno_width = fno_width
        self.fno_fc_units = fno_fc_units
        self.conf_interval_z = conf_interval_z
        self.onnx_opset = onnx_opset
        self.output_path = output_path

        self.mlp_ensemble = (
            make_ensemble_mlp(
                data_length=data_length,
                num_layers=mlp_num_layers,
                units_per_layer=mlp_units_per_layer,
                num_models=num_mlp,
                name="mlp_ensemble",
                dtype=dtype,
            )
            if num_mlp > 0
            else None
        )
        self.elm_ensemble = (
            make_ensemble_elm(
                data_length=data_length,
                units_per_layer=list(elm_units_per_layer),
                num_models=num_elm,
                name="elm_ensemble",
                dtype=dtype,
            )
            if num_elm > 0
            else None
        )
        self.fno_models = [
            FNO1d(
                data_length=data_length,
                modes=fno_modes,
                width=fno_width,
                fc_units=fno_fc_units,
                name=f"fno_{i}",
                dtype=dtype,
            )
            for i in range(num_fno)
        ]

        self.input_scale = tf.Variable(1.0, trainable=False, dtype=self._float_dtype, name="input_scale")
        self.output_scale = tf.Variable(1.0, trainable=False, dtype=self._float_dtype, name="output_scale")

        grid = np.linspace(0.0, 1.0, data_length).astype(np.float32 if dtype == "float32" else np.float64)
        self._fno_coords = tf.constant(grid, dtype=self._float_dtype)  # (data_length,)

        total_members = num_mlp + num_elm + num_fno
        self._member_weights = self.add_weight(
            name="member_weights", shape=(total_members,), initializer="zeros", trainable=False, dtype=self._float_dtype
        )
        self.set_family_weights(family_weights)

    def _compute_member_weights(self, family_weights):
        counts = {"mlp": self.num_mlp, "elm": self.num_elm, "fno": self.num_fno}
        np_dtype = np.float64 if self._float_dtype == tf.float64 else np.float32

        if family_weights is None:
            total_members = sum(counts.values())
            return np.full(total_members, 1.0 / total_members, dtype=np_dtype)

        weights = {k: float(family_weights.get(k, 1.0)) for k in counts}
        included_weight = sum(w for k, w in weights.items() if counts[k] > 0)
        if included_weight <= 0:
            raise ValueError("family_weights must assign a positive weight to at least one family with members > 0.")

        per_member = []
        for k in ("mlp", "elm", "fno"):
            n = counts[k]
            if n == 0:
                continue
            share = weights[k] / included_weight
            per_member.extend([share / n] * n)
        return np.array(per_member, dtype=np_dtype)

    def set_family_weights(self, family_weights=None):
        """
        Sets how much each model family contributes to the combined mean/std -- can be called on
        an already-trained model at any time, since combination weighting doesn't change what any
        sub-model learned, only how their predictions are combined.

        Args:
            family_weights (dict or None): An optional dict with any of "mlp", "elm", "fno" keys
                (missing keys default to 1.0), giving that family's total share of the combined
                prediction regardless of how many members it has -- e.g. `{"mlp": 1, "elm": 1}`
                gives the two families equal say even if `num_elm` is much larger than `num_mlp`.
                Pass None (the default) to weight every individual member equally instead, so each
                family's influence is simply proportional to its member count.
        """
        weights = self._compute_member_weights(family_weights)
        self._member_weights.assign(weights)
        self.family_weights = family_weights

    def _fno_input(self, x_scaled):
        coords = tf.broadcast_to(self._fno_coords[None, :], tf.shape(x_scaled))
        return tf.stack([x_scaled, coords], axis=-1)  # (batch, data_length, 2)

    def fit(self, X, Y, epochs=100, batch_size=64, alpha=0.0, verbose=1):
        """
        Trains every included sub-model on (X, Y) -- raw density -> raw emission -- handling each
        architecture's own preferred representation and training procedure internally:
        `EnsembleELM_Bayesian` is solved in closed form on raw data (as `neuro_bes.network.elm.train`
        does); `EnsembleMLP` and every `FNO1d` member are trained iteratively via backprop on
        [0, 1]-scaled data (as `neuro_bes.network.deepNN.train` does), using a shared Adam optimizer
        per sub-model. Returns a dict of that sub-model's training history/loss for each of "mlp",
        "elm", and "fno" (the latter a list, one per FNO member); a family left out via
        `num_mlp`/`num_elm`/`num_fno` = 0 is omitted from the returned dict.
        """
        X = np.asarray(X, dtype=np.float64 if self._float_dtype == tf.float64 else np.float32)
        Y = np.asarray(Y, dtype=np.float64 if self._float_dtype == tf.float64 else np.float32)

        input_scale = np.max(np.abs(X))
        output_scale = np.max(np.abs(Y))
        self.input_scale.assign(input_scale)
        self.output_scale.assign(output_scale)

        X_scaled = (X / input_scale).astype(X.dtype)
        Y_scaled = (Y / output_scale).astype(Y.dtype)

        histories = {}

        if self.elm_ensemble is not None:
            self.elm_ensemble.compile()
            self.elm_ensemble.fit(X, Y, batch_size=len(X), epochs=1, verbose=0, shuffle=False)
            histories["elm"] = "closed-form (one ridge-regression solve)"

        if self.mlp_ensemble is not None:
            self.mlp_ensemble.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
            mlp_history = self.mlp_ensemble.fit(X_scaled, Y_scaled, epochs=epochs, batch_size=batch_size, verbose=verbose)
            histories["mlp"] = mlp_history.history["loss"]

        if self.fno_models:
            X_fno_scaled = self._fno_input(tf.constant(X_scaled)).numpy()
            histories["fno"] = []
            for fno_model in self.fno_models:
                fno_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
                fno_history = fno_model.fit(X_fno_scaled, Y_scaled, epochs=epochs, batch_size=batch_size, verbose=verbose)
                histories["fno"].append(fno_history.history["loss"])

        return histories

    def call(self, inputs):
        x_raw = tf.convert_to_tensor(inputs, dtype=self._float_dtype)
        x_scaled = x_raw / self.input_scale
        pooled = []

        if self.mlp_ensemble is not None:
            pooled.append(self.mlp_ensemble._raw_predictions(x_scaled) * self.output_scale)  # (num_mlp, batch, data_length)
        if self.elm_ensemble is not None:
            pooled.append(self.elm_ensemble.raw_predictions(x_raw))  # (num_elm, batch, data_length), already raw units
        if self.fno_models:
            x_fno_scaled = self._fno_input(x_scaled)
            pooled.append(tf.stack([m(x_fno_scaled) for m in self.fno_models], axis=0) * self.output_scale)  # (num_fno, batch, data_length)

        all_raw = tf.concat(pooled, axis=0)  # (num_mlp+num_elm+num_fno, batch, data_length)
        weights = tf.reshape(self._member_weights, [-1, 1, 1])
        mean_pred = tf.reduce_sum(all_raw * weights, axis=0)
        variance = tf.reduce_sum(weights * tf.square(all_raw - mean_pred), axis=0)
        std_pred = tf.sqrt(variance)
        ci_lower = mean_pred - self.conf_interval_z * std_pred
        ci_upper = mean_pred + self.conf_interval_z * std_pred
        return tf.concat([mean_pred, std_pred, ci_lower, ci_upper], axis=1)

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
            "num_mlp": self.num_mlp,
            "num_elm": self.num_elm,
            "num_fno": self.num_fno,
            "mlp_num_layers": self.mlp_num_layers,
            "mlp_units_per_layer": self.mlp_units_per_layer,
            "elm_units_per_layer": self.elm_units_per_layer,
            "fno_modes": self.fno_modes,
            "fno_width": self.fno_width,
            "fno_fc_units": self.fno_fc_units,
            "conf_interval_z": self.conf_interval_z,
            "family_weights": self.family_weights,
            "onnx_opset": self.onnx_opset,
            "output_path": self.output_path,
            "dtype": _DTYPE_NAME_MAP[self._float_dtype],
        })
        return config


def make_mixed_ensemble(
    data_length,
    num_mlp=10,
    num_elm=10,
    num_fno=5,
    mlp_num_layers=4,
    mlp_units_per_layer=32,
    elm_units_per_layer=(200, 200),
    fno_modes=10,
    fno_width=32,
    fno_fc_units=64,
    conf_interval_z=1.96,
    family_weights=None,
    name="MixedEnsemble",
    onnx_opset=13,
    output_path="model.onnx",
    dtype="float32",
):
    """
    Builds a heterogeneous ensemble of `num_mlp` MLPs, `num_elm` ELM members, and `num_fno` FNOs.
    Call `.fit(X, Y, epochs=..., batch_size=...)` on the result before using it for predictions --
    see `MixedEnsemble` for how each sub-model's own training procedure and input convention is
    handled internally.

    Args:
        data_length (int): Length of the input/output profile.
        num_mlp (int): Number of MLP ensemble members. May be 0 to leave MLP out entirely.
        num_elm (int): Number of ELM ensemble members. May be 0 to leave ELM out entirely.
        num_fno (int): Number of FNO ensemble members. May be 0 to leave FNO out entirely. Not
            internally vectorized (no batched-ensemble FNO exists), so cost scales roughly
            linearly with this when > 0 -- keep it modest. At least one of num_mlp/num_elm/num_fno
            must be positive.
        mlp_num_layers, mlp_units_per_layer: Passed to `make_ensemble_mlp`.
        elm_units_per_layer: Passed to `make_ensemble_elm`.
        fno_modes, fno_width, fno_fc_units: Passed to each `FNO1d` member.
        conf_interval_z (float): Z-score for the reported confidence interval.
        family_weights (dict or None): How much each included family contributes to the combined
            mean/std -- see `MixedEnsemble.set_family_weights`. None (the default) weights every
            individual member equally; can be changed later via `set_family_weights` without
            retraining.
        name (str): Model name.
        onnx_opset (int): ONNX opset version for export.
        output_path (str): Default ONNX export path.
        dtype (str): Floating-point precision -- 'float32' or 'float64'.

    Returns:
        MixedEnsemble: The built model, not yet fit.
    """
    return MixedEnsemble(
        data_length=data_length,
        num_mlp=num_mlp,
        num_elm=num_elm,
        num_fno=num_fno,
        mlp_num_layers=mlp_num_layers,
        mlp_units_per_layer=mlp_units_per_layer,
        elm_units_per_layer=elm_units_per_layer,
        fno_modes=fno_modes,
        fno_width=fno_width,
        fno_fc_units=fno_fc_units,
        conf_interval_z=conf_interval_z,
        family_weights=family_weights,
        name=name,
        onnx_opset=onnx_opset,
        output_path=output_path,
        dtype=dtype,
    )
