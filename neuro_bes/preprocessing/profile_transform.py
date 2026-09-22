import numpy as np
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d
from sklearn.pipeline import Pipeline

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
    
class DensityScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.global_max_ = None

    def fit(self, batch):
        if self.global_max_ is not None:
            raise RuntimeError("The transformer has already been fitted.")
        all_maxes = [
            np.max(bes_obj.densities) 
            for bes_obj in batch 
            if bes_obj.densities is not None
            ]
        if not all_maxes:
            raise ValueError("No density data to fit.")
        self.global_max_ = np.max(all_maxes)
        if not np.isfinite(self.global_max_):
            raise ValueError("Global maximum is not finite, cannot scale data.")
        if self.global_max_ == 0:
            raise ValueError("Global maximum is zero, cannot scale data.")
        return self

    def transform(self, batch):
        if self.global_max_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.densities is not None:
                bes_obj.densities = bes_obj.densities / self.global_max_
        return batch_copy

    def inverse_transform(self, batch):
        if self.global_max_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.densities is not None and bes_obj.densities.size > 0:
                bes_obj.densities = bes_obj.densities * self.global_max_
        return batch_copy

class EmissionScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.global_max_ = None

    def fit(self, batch):
        if self.global_max_ is not None:
            raise RuntimeError("The transformer has already been fitted.")
        all_maxes = [
            np.max(bes_obj.emissions) 
            for bes_obj in batch 
            if bes_obj.emissions is not None and bes_obj.emissions.size > 0
            ]
        if not all_maxes:
            raise ValueError("No emission data to fit.")
        self.global_max_ = np.max(all_maxes)
        if not np.isfinite(self.global_max_):
            raise ValueError("Global maximum is not finite, cannot scale data.")
        if self.global_max_ == 0:
            raise ValueError("Global maximum is zero, cannot scale data.")
        return self

    def transform(self, batch):
        if self.global_max_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.emissions is not None:
                bes_obj.emissions = bes_obj.emissions / self.global_max_
        return batch_copy

    def inverse_transform(self, batch):
        if self.global_max_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.emissions is not None:
                bes_obj.emissions = bes_obj.emissions * self.global_max_
        return batch_copy

class BeamIntensityScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.integral_emissions_ = []

    def fit(self, batch):
        """
        This method is present to conform to the scikit-learn transformer interface; it does nothing with the input batch.
        """
        return self
    def transform(self, batch):
        """
        Normalizes each emission profile in the batch by its total area, calculated using numerical integration and an extrapolated tail if needed.
        Expects a batch of objects with .emission' and '.grid' attributes; returns the batch with normalized emission profiles.
        """
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.emissions is not None and bes_obj.emissions.size > 0:
                if bes_obj.grid is None:
                    raise ValueError(f"Grid is None for object with ID {bes_obj.ID}")
                total_area=[]
                for idx, emission in enumerate(bes_obj.emissions):
                    if bes_obj.grid[0]>bes_obj.grid[-1]:
                        grid=bes_obj.grid[::-1]
                        em=emission[::-1]
                    else:
                        grid=bes_obj.grid
                        em=emission
                    area_under_profile=np.trapz(em,x=grid)
                    if em[0]>np.max(em)/2:
                        log_em = np.log(em[:10])
                        coeffs = np.polyfit(grid[:10], log_em, 1)
                        k = coeffs[0]
                        A = np.exp(coeffs[1])
                        extra_area = (A / k) * np.exp(k * grid[0])
                        if np.isnan(extra_area) or np.isinf(extra_area):
                            extra_area = 0
                            #print("Warning: Extrapolated area is NaN or Inf, setting extra_area to 0. Check the " + str(idx) + ". emission profile for object ID " + str(bes_obj.ID))
                        if extra_area < 0:
                            extra_area = 0
                            #print("Warning: Negative extrapolated area calculated, setting extra_area to 0. Check the " + str(idx) + ". emission profile for object ID " + str(bes_obj.ID))
                        #if extra_area > area_under_profile:
                        #    raise ValueError(f"Extrapolated tail area ({extra_area:.2e}) is larger than the area under the profile ({area_under_profile:.2e})")
                        area_tot = area_under_profile + extra_area
                        if area_tot< 1:
                            area_tot = 1
                            #print("Warning: Total area under the emission profile is less than 1, setting it to 1. Check the " + str(idx) + ". emission profile for object ID " + str(bes_obj.ID))
                        total_area.append(area_tot)
                    else:
                        if area_under_profile < 1:
                            area_under_profile = 1
                            #print("Warning: Total area under the emission profile is less than 1, setting it to 1. Check the " + str(idx) + ". emission profile for object ID " + str(bes_obj.ID))
                        total_area.append(area_under_profile)
                total_area=np.array(total_area)
                if np.any(total_area == 0):
                    raise ValueError(f"Total area under the emission profile is zero for object ID {bes_obj.ID}, cannot normalize.")
                bes_obj.emissions=bes_obj.emissions/total_area[:,None]
                self.integral_emissions_.append({'id':bes_obj.ID, 'area':total_area})
        return batch_copy
    
    def inverse_transform(self, batch):
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.emissions is not None:
                area = next((item['area'] for item in self.integral_emissions_ if item['id'] == bes_obj.ID), None)
                if area is not None:
                    bes_obj.emissions = bes_obj.emissions * area[:,None]
                else:
                    raise ValueError(f"No integral emission area found for object with ID {bes_obj.ID} during inverse transformation.")
        return batch_copy
    
class DensityFlatout1(BaseEstimator, TransformerMixin):
    def fit(self, batch):
        return self
    
    def transform(self, batch):
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.densities is not None and bes_obj.emissions is not None:
                if bes_obj.grid is None:
                    raise ValueError(f"Grid is None for object with ID {bes_obj.ID}")
                for emission,density in zip(bes_obj.emissions, bes_obj.densities):
                    threshold = 0.1 * np.max(emission)
                    below = emission < threshold
                    # Find trailing True values
                    if bes_obj.grid[0]>bes_obj.grid[-1]:
                        below=below[::-1]
                    count = np.argmax(~below) if np.any(~below) else bes_obj.resolution
                    if count > 0 and count < bes_obj.resolution:
                        if bes_obj.grid[0]>bes_obj.grid[-1]:
                            density[-count:] = density[-count - 1]
                        else:
                            density[:count] = density[count]
        return batch_copy

class DensityFlatout2(BaseEstimator, TransformerMixin):
    def fit(self, batch):
        return self
    
    def transform(self, batch):
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.densities is not None and bes_obj.emissions is not None:
                if bes_obj.grid is None:
                    raise ValueError(f"Grid is None for object with ID {bes_obj.ID}")
                for emission,density in zip(bes_obj.emissions, bes_obj.densities):
                    # find where cumulated emission exceeds 90% of total
                    if bes_obj.grid[0]<bes_obj.grid[-1]:
                        em=emission[::-1]
                        grid=bes_obj.grid[::-1]
                    else:
                        em=emission
                        grid=bes_obj.grid
                    # find the cumulative sum taking into account the grid spacing
                    cumulative = np.cumsum(em * np.abs(np.gradient(grid)))
                    threshold = 0.9
                    idx = np.searchsorted(cumulative, threshold)
                    if idx < len(em):
                        idx=len(em)-idx
                        if bes_obj.grid[0]>bes_obj.grid[-1]:
                            if idx>=len(density):
                                print(bes_obj.ID)
                                print(em)
                                print(density)
                            density[-idx:] = density[-idx - 1]
                        else:
                            density[:idx] = density[idx]
        return batch_copy 
    
class InterpolateToCommonGrid(BaseEstimator, TransformerMixin):
    def __init__(self, n_points=40, kind="linear"):
        self.n_points = n_points
        self.kind = kind
        self.original_grids_ = []
        self.common_grid_ = None

    def fit(self, batch):
        if self.common_grid_ is not None:
            raise RuntimeError("The transformer has already been fitted.")
        all_grids = [bes_obj.grid for bes_obj in batch if bes_obj.grid is not None]
        if not all_grids:
            raise ValueError("No grid data to fit.")
        gridmax = np.max(np.concatenate(all_grids))
        gridmin = np.min(np.concatenate(all_grids))
        self.common_grid_ = np.linspace(gridmax, gridmin, self.n_points)
        return self

    def transform(self, batch):
        if self.common_grid_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        self.original_grids_.extend([{'id':bes_obj.ID, 'grid':bes_obj.grid} for bes_obj in batch if bes_obj.grid is not None])
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            if bes_obj.grid is not None and bes_obj.emissions is not None and bes_obj.emissions.size > 0:
                emission_newgrid = np.stack([
                    interp1d(bes_obj.grid, emission, kind=self.kind)(self.common_grid_)
                    for emission in bes_obj.emissions
                    ])
                density_newgrid = None
                if bes_obj.densities is not None and bes_obj.densities.size > 0:
                    density_newgrid = np.stack([
                    interp1d(bes_obj.grid, density, kind=self.kind)(self.common_grid_)
                    for density in bes_obj.densities
                    ])
                bes_obj.grid = self.common_grid_
                bes_obj.emissions = emission_newgrid
                bes_obj.densities = density_newgrid
        return batch_copy

    def inverse_transform(self, batch):
        if self.original_grids_ is None:
            raise RuntimeError("The data has not been transformed yet.")
        batch_copy = copy.deepcopy(batch)
        for bes_obj in batch_copy:
            # find the original grid in self.original_grids_ by matching the ID
            original_grid = next((item['grid'] for item in self.original_grids_ if item['id'] == bes_obj.ID), None)
            if original_grid is not None and bes_obj.emissions is not None and bes_obj.emissions.size > 0 and bes_obj.densities is not None and bes_obj.densities.size > 0:
                emission_oldgrid = np.stack([
                interp1d(self.common_grid_, emission, kind=self.kind, fill_value="extrapolate")(original_grid)
                for emission in bes_obj.emissions])
                density_oldgrid = np.stack([
                interp1d(self.common_grid_, density, kind=self.kind, fill_value="extrapolate")(original_grid)
                for density in bes_obj.densities])
                bes_obj.grid = original_grid
                bes_obj.emissions = emission_oldgrid
                if bes_obj.densities is not None and bes_obj.densities.size > 0:
                    densities_oldgrid = np.stack([
                    interp1d(self.common_grid_, density, kind=self.kind, fill_value="extrapolate")(original_grid)
                    for density in bes_obj.densities])
                    bes_obj.densities = densities_oldgrid
        return batch_copy

class PartialInversePipeline(Pipeline):
    def inverse_transform(self, batch):
        for _, step in reversed(self.steps):
            if hasattr(step, "inverse_transform"):
                batch = step.inverse_transform(batch)
            else:
                # make a warning that this step does not have inverse_transform and is being skipped
                print(f"Warning: Step '{_}' does not have an inverse_transform method and will be skipped in the inverse transformation.")
        return batch