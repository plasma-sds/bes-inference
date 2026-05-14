import os
import numpy as np
import h5py
from scipy.interpolate import interp1d

from neuro_bes.data import besInferenceDatapoints

def load_w7x_synthetic_data(file_path, bes_obj_id):
    """
    Loads synthetic W7X data from an HDF5 file and returns a besInferenceDatapoints object.
    The HDF5 file is expected to have a dataset called "/init_data" with the following structure:
        - The first 8 columns are ignored.
        - The next 1000 columns (8:1008) contain the r coordinates.
        - The next 1000 columns (1008:2008) contain the densities.
        - The next 1000 columns (2008:3008) contain the emissions.
    The function will filter out rows where the r coordinates do not span at least 0.2m, and will interpolate the densities
     and emissions onto a common grid defined by the minimum and maximum. of the r coordinates across all valid rows. 
    The function also extracts metadata from the filename and stores it in the besInferenceDatapoints object. The filename is expected to have the following format:
    "W7Xinit-{species}-E_{energy}-t_{temperature}-q_{q}-zeff_{z_eff}-[...].hdf5

    Parameters:
    ----------
    file_path : str
        The path to the HDF5 file containing the synthetic W7X data.
    bes_obj_id : str
        The ID to assign to the besInferenceDatapoints object.
        
    Returns:
    -------
    besInferenceDatapoints
        An object containing the loaded and processed data, along with metadata extracted from the filename.
    """
    with h5py.File(file_path) as f:
        d = f["/init_data"][()]
        r_coord = d[:,8:1008]
        densities = d[:,1008:2008]
        emissions = d[:,2008:3008]
    print(r_coord)
    # delete data where r_coord does not span minimum 0.2m
    valid_indices = [i for i in range(r_coord.shape[0]) if np.max(r_coord[i]) - np.min(r_coord[i]) >= 0.2]
    r_coord = r_coord[valid_indices]
    densities = densities[valid_indices]
    emissions = emissions[valid_indices]
    # get the minimum of the maximum of r_coord's rows
    gridmax=min([np.max(r_coord[i]) for i in range(r_coord.shape[0])])
    gridmin=min([np.min(r_coord[i]) for i in range(r_coord.shape[0])])
    default_grid=np.linspace(gridmin,gridmax,1000)
    emission=[]
    density=[]
    for r,em,den in zip(r_coord,emissions,densities):
        f_interp = interp1d(r, em, kind='linear', fill_value="extrapolate")
        emission.append(f_interp(default_grid))
        f_interp = interp1d(r, den, kind='linear', fill_value="extrapolate")
        density.append(f_interp(default_grid))

    density=np.array(density)
    emission=np.array(emission)

    filename = os.path.basename(file_path)
    # get species, energy, temperature, q, z_eff from filename
    # species is the first part of the filename after "W7Xinit-" and before the first "-"
    species = filename.split("W7Xinit-")[1].split("-")[0]
    # energy is the part after "E_" and before the next "-"
    energy = filename.split("E_")[1].split("-")[0]
    # temperature is the part after "t_" and before the next "-"
    temperature = filename.split("t_")[1].split("-")[0]
    # q is the part after "q_" and before the next "-"
    q = filename.split("q_")[1].split("-")[0]
    # z_eff is the part after "zeff_" and before the next "-"
    z_eff = filename.split("zeff_")[1].split("-")[0]
    verbose="data loaded from "+file_path
    tags=[str(i) for i in range(density.shape[0])]

    bes_data=besInferenceDatapoints(grid=default_grid,energy=energy,species=species,ID=bes_obj_id,zeff=z_eff,q=q,temperature=temperature,verbose=verbose)
    print(default_grid)
    print("size of grid: ", bes_data.grid.shape)
    print("size of density: ", density.shape)
    print("size of emission: ", emission.shape)
    bes_data.add_datapoints_bulk(density, emission, tags)

    return bes_data