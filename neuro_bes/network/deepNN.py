import numpy as np
import tensorflow as tf

def make_MLP(
    data_length,
    num_layers=2,
    units_per_layer=64,
    activations='relu',
    output_activation=None,
    dropout_rate=None,
    use_batchnorm=False
):
    """
    Builds a customizable MLP model with input/output of the same length.

    Args:
        data_length (int): Length of the input/output data.
        num_layers (int): Number of hidden layers.
        units_per_layer (int or list): Number of neurons per layer. If int, same for all layers.
        activations (str or list): Activation(s) for each layer. If str, same for all layers.
        output_activation (str or None): Activation for output layer.
        dropout_rate (float or None): Dropout rate (if any).
        use_batchnorm (bool): Whether to use BatchNormalization after each layer.

    Returns:
        tf.keras.Model: The built MLP model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(data_length,)))

    # Prepare units and activations as lists
    if isinstance(units_per_layer, int):
        units = [units_per_layer] * num_layers
    else:
        units = units_per_layer
    if isinstance(activations, str):
        acts = [activations] * num_layers
    else:
        acts = activations
        
    if len(units) != num_layers or len(acts) != num_layers:
        raise ValueError("Length of units_per_layer and activations must match num_layers.")

    for u, a in zip(units, acts):
        model.add(tf.keras.layers.Dense(u, activation=a))
        if use_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout_rate is not None:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(data_length, activation=output_activation))
    return model

def make_CNN(
    data_length,
    num_layers=2,
    filters_per_layer=[16,32,32],
    kernel_size_per_layer=10,
    activations='relu',
    output_activation='linear',
    dropout_rate=0.2,
    use_batchnorm=False
):
    """
    Builds a customizable CNN model with input/output of the same length.

    Args:
        data_length (int): Length of the input/output data.
        num_layers (int): Number of hidden layers.
        filters_per_layer (int or list): Number of filters in the convolution per layer. If int, same for all layers.
        kernel_size_per_layer (int or list): Size of the convolution window per layer. If int, same for all layers.
        activations (str or list): Activation(s) for each layer. If str, same for all layers.
        output_activation (str or None): Activation for output layer.
        dropout_rate (float or None): Dropout rate (if any).
        use_batchnorm (bool): Whether to use BatchNormalization after each layer.

    Returns:
        tf.keras.Model: The built CNN model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(data_length,1)))

    # Prepare filters, kernel_sizes and activations as lists
    if isinstance(filters_per_layer, int):
        filters = [filters_per_layer] * num_layers
    else:
        filters = filters_per_layer
    if isinstance(kernel_size_per_layer, int):
        kernel_sizes = [kernel_size_per_layer] * num_layers
    else:
        kernel_sizes = kernel_size_per_layer
    if isinstance(activations, str):
        acts = [activations] * num_layers
    else:
        acts = activations
        
    if len(filters) != num_layers or len(kernel_sizes) != num_layers or len(acts) != num_layers:
        raise ValueError("Length of units_per_layer and activations must match num_layers.")

    for f, ks, a in zip(filters, kernel_sizes, acts):
        model.add(tf.keras.layers.Conv1D(f, kernel_size=ks, activation=a, padding='same'))
        if use_batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout_rate is not None:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Conv1D(1, kernel_size=1, activation=output_activation, padding='same'))
    return model

def train(model, x, y, epochs=100, batch_size=64, weight_samples=False):
    checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    split_idx = int(0.75 * x.shape[0])
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    AUTOTUNE = tf.data.AUTOTUNE
    if weight_samples:
        y_train_max = np.max(y_train, axis=1)
        hist, bin_edges = np.histogram(y_train_max, bins='auto')
        # Avoid zero counts for inverse
        hist = np.where(hist == 0, 1, hist)
        # Assign each sample to a bin
        bin_indices = np.digitize(y_train_max, bin_edges[:-1], right=True)
        # Map each sample to its bin's count
        bin_counts = hist[bin_indices - 1]
        # Inverse proportional weights
        sample_weights = 1.0 / bin_counts
        sample_weights = sample_weights.astype(np.float32)
        # Normalize weights to mean 1
        sample_weights /= np.mean(sample_weights)
        # allow a maximum of 10 for weights
        sample_weights = np.clip(sample_weights, 0, 10)
        # Do the same for validation set
        y_val_max = np.max(y_val, axis=1)
        bin_indices_val = np.digitize(y_val_max, bin_edges[:-1], right=True)
        bin_counts_val = hist[bin_indices_val - 1]
        val_sample_weights = 1.0 / bin_counts_val
        val_sample_weights = val_sample_weights.astype(np.float32)
        val_sample_weights /= np.mean(val_sample_weights)
        val_sample_weights = np.clip(val_sample_weights, 0, 10)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weights))
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val, val_sample_weights))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    train_ds = (train_ds.shuffle(buffer_size=10000).batch(batch_size).prefetch(buffer_size=AUTOTUNE))
    val_ds = (val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, batch_size=batch_size, callbacks=[model_checkpoint_callback])
    model=tf.keras.models.load_model(checkpoint_filepath)
    return model, history