import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class GlobalMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scales data by dividing by the global maximum value across all features and samples.
    """
    def __init__(self):
        self.global_max_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.global_max_ = np.max(X)
        if not np.isfinite(self.global_max_):
            raise ValueError("Global maximum is not finite, cannot scale data.")
        if self.global_max_ == 0:
            raise ValueError("Global maximum is zero, cannot scale data.")
        return self

    def transform(self, X):
        if self.global_max_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        X = np.asarray(X)
        return X / self.global_max_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        if self.global_max_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        X = np.asarray(X)
        return X * self.global_max_

class CenterMaxTransformer(BaseEstimator, TransformerMixin):
    """
    Shifts each row of a 2D numpy array so that the maximum value is centered.
    Pads as needed to preserve all values. Supports inverse transformation.
    The new length is symmetric: orig_len + abs(min_shift) + abs(max_shift).
    """
    def __init__(self):
        self.shifts_ = None
        self.orig_len_ = None
        self.new_len_ = None
        self.left_pad_ = None
        self.right_pad_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        self.orig_len_ = X.shape[1]
        center_idx = self.orig_len_ // 2
        max_indices = np.argmax(X, axis=1)
        self.shifts_ = center_idx - max_indices
        min_shift = np.min(self.shifts_)
        max_shift = np.max(self.shifts_)
        self.left_pad_ = abs(min_shift)
        self.right_pad_ = abs(max_shift)
        self.new_len_ = self.orig_len_ + self.left_pad_ + self.right_pad_
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.shifts_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        n_samples = X.shape[0]
        X_shifted = np.zeros((n_samples, self.new_len_), dtype=X.dtype)
        for i in range(n_samples):
            shift = self.shifts_[i]
            insert_start = self.left_pad_ + shift
            insert_end = insert_start + self.orig_len_
            X_shifted[i, insert_start:insert_end] = X[i]
            # Pad left if needed
            if insert_start > 0:
                if self.orig_len_ > 1:
                    slope = X[i,1] - X[i,0]
                    for j in range(insert_start-1, -1, -1):
                        val = X[i,0] + slope * (j - (insert_start-1))
                        X_shifted[i, j] = max(val, 0)
                else:
                    X_shifted[i, :insert_start] = max(X[i,0], 0)
            # Pad right if needed
            if insert_end < self.new_len_:
                if self.orig_len_ > 1:
                    slope = X[i,-1] - X[i,-2]
                    for j in range(insert_end, self.new_len_):
                        val = X[i,-1] + slope * (j - insert_end + 1)
                        X_shifted[i, j] = max(val, 0)
                else:
                    X_shifted[i, insert_end:] = max(X[i,-1], 0)
        return X_shifted

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_shifted):
        X_shifted = np.asarray(X_shifted)
        if self.shifts_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        n_samples = X_shifted.shape[0]
        X_orig = np.zeros((n_samples, self.orig_len_), dtype=X_shifted.dtype)
        for i in range(n_samples):
            shift = self.shifts_[i]
            insert_start = self.left_pad_ + shift
            insert_end = insert_start + self.orig_len_
            X_orig[i] = X_shifted[i, insert_start:insert_end]
        return X_orig
