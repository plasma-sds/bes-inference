import tensorflow as tf
import logging


class LOSSES():
    

    
    def map_losses_and_metrics(self, kwarg : str):
        
        metric_loss_map = {"eval_huber":self.eval_huber,
                    "eval_mse":self.eval_mse,
                    "eval_mae":self.eval_mae,
                    "eval_mape":self.eval_mape,
                    "eval_log_cosh":self.eval_log_cosh,
                    "eval_binary_crossentropy":self.eval_binary_crossentropy,
                    "eval_categorial_crossentropy":self.eval_categorial_crossentropy,
                    "eval_kl_divergence":self.eval_kl_divergence
                    }
        
        return metric_loss_map[kwarg]
        
        
    def eval_mse(self, y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(y_true - y_pred)) 
    
    def eval_mape(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), 1e-7))  # Avoid division by zero
        return tf.reduce_mean(error) * 100.0

    def eval_mae(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    def eval_huber(self, y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 1.0) -> tf.Tensor:
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic_part = tf.minimum(abs_error, delta)
        linear_part = abs_error - quadratic_part
        return tf.reduce_mean(0.5 * tf.square(quadratic_part) + delta * linear_part)
    
    def eval_log_cosh(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.math.log(tf.cosh(y_true - y_pred)))
    
    def eval_binary_crossentropy(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    
    def eval_categorial_crossentropy(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    
    def eval_kl_divergence(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(y_true, y_pred))







    
    