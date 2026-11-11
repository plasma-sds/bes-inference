import tensorflow as tf
import tf2onnx
import onnx

_DTYPE_MAP = {
    "float32": tf.float32,
    "float64": tf.float64,
}


def export_to_onnx(model, input_shape, output_path="model.onnx", opset=13, dtype="float32"):
    """
    Exports a Keras model (Sequential or subclassed) to ONNX.

    tf2onnx.convert.from_keras fails on plain tf.keras.Sequential models with a KeyError in its
    output-name lookup and relies on traced model inputs, so this wraps the model call in a
    tf.function and uses tf2onnx.convert.from_function instead, which works for any model.

    Args:
        model (tf.keras.Model): A trained model.
        input_shape (tuple): The model's input shape excluding the batch dimension, e.g.
            (data_length,) for make_MLP / make_ensemble_elm or (data_length, 1) for make_CNN.
        output_path (str): Path to write the ONNX file to.
        opset (int): ONNX opset version for export.
        dtype (str or tf.DType): Floating-point precision of the model's input -- 'float32' or 'float64'.

    Returns:
        str: The output_path the model was written to.
    """
    if isinstance(dtype, str):
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype must be 'float32' or 'float64', got '{dtype}'")
        dtype = _DTYPE_MAP[dtype]
    input_signature = [tf.TensorSpec([None, *input_shape], dtype, name="input")]
    call_fn = tf.function(lambda x: model(x, training=False), input_signature=input_signature)
    onnx_model, _ = tf2onnx.convert.from_function(call_fn, input_signature=input_signature, opset=opset)
    onnx.save_model(onnx_model, output_path)
    return output_path
