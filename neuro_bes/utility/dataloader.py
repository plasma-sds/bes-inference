import os
import numpy as np
import h5py
from scipy.interpolate import interp1d
import pandas as pd

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
    bes_data.add_datapoints_bulk(density, emission, tags)
    print("Created besInferenceDatapoints object with ID:", bes_obj_id, "species:", species, "energy:", energy, "temperature:",
           temperature, "q:", q, "z_eff:", z_eff, "grid shape:", default_grid.shape, "density shape:", density.shape, "emission shape:", emission.shape)

    return bes_data

def load_asdex_experimental_data(file_path, bes_obj_id, exclude=[]):
    """
    Loads experimental ASDEX data from an HDF5 file and returns a besInferenceDatapoints object. 
    The function will filter out shots which have different grid from the first shot encountered during loading.
    Unknown species, energy, temperature, q, and z_eff are set to NaN. 

    Parameters:
    ----------
    file_path : str
        The path to the HDF5 file containing the experimental ASDEX data.
    bes_obj_id : str
        The ID to assign to the besInferenceDatapoints object.
    exclude : list of str, optional
        A list of shot names to exclude from loading. Default is an empty list.
        
    Returns:
    -------
    besInferenceDatapoints
        An object containing the loaded and processed data.
    """
    with h5py.File(file_path) as f:
        emission = []
        density = []
        r_coord = []
        shots = list(f.keys())
        print("Shots found in the HDF5 file:", shots)
        print("Shots skipped:", exclude)
        for shot in shots:
            if shot in exclude:
                pass
            else:
                r_coord.append(f[shot]["x2"][:])
                emission.append(f[shot]["lib2mod"][:])
                density.append(f[shot]["ne"][:])

    

    r_coord=np.vstack(r_coord[:])
    print("Grid of the different shots:", r_coord)
    mask_samegrid=np.all(r_coord[0]==r_coord,axis=1)
    print("Mask applied for shots with same grid: ", mask_samegrid)
    emission=[arr for arr, keep in zip(emission[:], mask_samegrid) if keep] #masking out shots with different grids
    density=[arr for arr, keep in zip(density[:], mask_samegrid) if keep] #masking out shots with different grids
    density=np.vstack(density[:]) 
    emission=np.vstack(emission[:])
    
    filename = os.path.basename(file_path)
    grid = r_coord[0]
    # get species, energy, temperature, q, z_eff
    species = "Li"
    # energy is
    energy = np.nan
    # temperature is
    temperature = np.nan
    # q is
    q = np.nan
    # z_eff is
    z_eff = np.nan


    verbose="data loaded from "+file_path
    tags=[str(i) for i in range(density.shape[0])]

    bes_data=besInferenceDatapoints(grid=grid,energy=energy,species=species,ID=bes_obj_id,zeff=z_eff,q=q,temperature=temperature,verbose=verbose)
    bes_data.add_datapoints_bulk(density, emission, tags)
    print("Created besInferenceDatapoints object with ID:", bes_obj_id, "species:", species, "energy:", energy, "temperature:", temperature, "q:", q, "z_eff:", z_eff, 
          "grid shape:", grid.shape, "density shape:", density.shape, "emission shape:", emission.shape)

    return bes_data

def load_tanh_synthetic_data(file_path, kind="static"):
    """
    Loads static or fluctuation synthetic data from HDF5 files and returns a list of besInferenceDatapoints objects. 
    The HDF5 files are expected to have a dataset called "df" with the following columns:
        - "T shape[eV]": The temperature shape in eV.
        - "Emission 3p-->3s" or "Emission 2p-->2s", "Emission 3n-->2n" etc: The emission data.
        - "Density Shape[$1/m^3$]": The density data.
    The function will group the data by unique temperature profiles and create a besInferenceDatapoints object for each unique temperature profile. 
    The grid is defined as a linear space from 0m to 0.4m with the same number of points as the density and emission data. 
    The species is extracted from the filename by taking the part before the first underscore, e.g. "Li" from "Li_statdataset.hdf5".
    The ID is generated as "dss_{index}" or "dsf_{index}" where {index} is the index of the unique temperature profile. 
    The verbose description is set to "static/fluctuation synthetic data on a grid with 1mm resolution". 
    The function returns a list of besInferenceDatapoints objects, one besInferenceDatapoints for each unique temperature profile.
    Parameters:
    ----------
    file_path : str
        The path to the directory containing the HDF5 files with the static or fluctuation synthetic data.
    kind : str, optional
        The type of data to load. Can be "static" or "fluctuation". Default is "static".
    Returns:
    -------
    list of besInferenceDatapoints
        A list of objects containing the loaded and processed data, one for each unique temperature profile.
    """
    filenames=[i for i in os.listdir(file_path) if i.endswith(".hdf5")]
    df=pd.DataFrame()
    for filename in filenames:
        df=pd.concat([df, pd.read_hdf(os.path.join(file_path, filename), "df")])
    df['T_key'] = df['T shape[eV]'].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else (x,))
    unique_T = df['T_key'].unique()
    groups = dict(tuple(df.groupby('T_key')))
    bes_data_batch=[]
    species=filename.split('_')[0]
    for ind,T in enumerate(unique_T):
        emission=[]
        if species=="Na":
            [emission.append(i) for i in groups[T]['Emission 3p-->3s'].values]
        elif species=="Li":
            [emission.append(i) for i in groups[T]['Emission 2p-->2s'].values]
        elif species=="D":
            [emission.append(i) for i in groups[T]['Emission 3n-->2n'].values]
        else:
            raise ValueError(f"Unknown species {species}. Beam species must be either 'Na', 'Li' or 'D'")
        emission=np.array(emission)
        density=[]
        [density.append(i) for i in groups[T]['Density Shape[$1/m^3$]'].values]
        density=np.array(density)
        if len(density.shape) < 2:
            # skip this iteration if density is 1D or 0D
            continue
        grid=np.linspace(0,0.4,density.shape[1])
        # get the species from the filename by going to the first underscore
        temperature=groups[T]['T shape[eV]'].values[0]
        # check if "temperature" is a string, if so, convert it to a float array, e.g. '[45, 34, 21]' -> np.array([45, 34, 21])
        if isinstance(temperature, str):
            temperature = np.fromstring(temperature.strip().strip("[]"), sep=",", dtype=np.float32)
        tags=[str(i) for i in range(density.shape[0])]
        if kind=="static":
            id="dss_"+str(ind)
            verbose="static synthetic data on a grid with 1mm resolution"
        elif kind=="fluctuation":
            id="dsf_"+str(ind)
            verbose="fluctuation synthetic data on a grid with 1mm resolution"
        else:
            raise ValueError(f"Unknown kind {kind}. Kind must be either 'static' or 'fluctuation'")
        bes_data=besInferenceDatapoints(grid=grid,species=species,ID=id,temperature=temperature,verbose=verbose)
        bes_data.add_datapoints_bulk(density, emission, tags)
        print("Created besInferenceDatapoints object with ID:", id, "species:", species, 
              "temperature shape:", temperature.shape, "grid shape:", grid.shape, "density shape:", density.shape, "emission shape:", emission.shape)    
        bes_data_batch.append(bes_data)
    return bes_data_batch


def load_hesel_synthetic_data(file_path, bes_obj_id, kind="fast"):
    """
    Loads HESEL synthetic data from HDF5 files and returns a besInferenceDatapoints object. 
    Takes a directory path containing the HESEL synthetic data files and loads all files of the specified kind ("fast" or "slow").
    The function will filter out files that do not match the specified kind and will raise a ValueError if the kind is not "fast" or "slow". 
    The function will concatenate the density and emission data from all valid files and create a besInferenceDatapoints object 
    with the concatenated data. 
    The species, energy, grid are extracted from the first valid file encountered during loading 
    and stored in the besInferenceDatapoints object. Any other hdf5 content (from further files) attributed differently will raise a ValueError. 
    The temperature is averaged across all valid files and the average temperature profile will be stored in the besInferenceDatapoints object.
    The ID is set to the provided bes_obj_id.


    Parameters:
    ----------
    file_path : str
        The path to the directory containing the HESEL synthetic data files.
    bes_obj_id : str
        The ID to assign to the created besInferenceDatapoints object.
    kind : str, optional
        The type of data to load. Can be "fast" or "slow". Default is "fast".
    Returns:
    -------
    besInferenceDatapoints
        An object containing the loaded data.
    """
    filenames=os.listdir(file_path)
    if kind=="fast":
        filenames=[i for i in filenames if (i.endswith(".h5") and "fast" in i)]
    elif kind=="slow":
        filenames=[i for i in filenames if (i.endswith(".h5") and "fast" not in i)]
    else:
        raise ValueError(f"Unknown kind {kind}. Kind must be either 'fast' or 'slow'")
    for ind,filename in enumerate(filenames):
        with h5py.File(os.path.join(file_path, filename), 'r') as f:
            if ind==0:
                energy = f['Beam energy [keV]'][()]
                species=f['Beam type'][()].decode("utf-8")
                grid=f['Grid [m]'][()]
                density=np.empty((0,grid.shape[0]))
                emission=np.empty((0,grid.shape[0]))
                temperature=np.empty((0,grid.shape[0]))     
            else:
                if f['Beam energy [keV]'][()] != energy:
                    raise ValueError(f"Energy mismatch: {f['Beam energy [keV]'][()]} keV != {energy} keV")
                if f['Beam type'][()].decode("utf-8") != species:
                    raise ValueError(f"Species mismatch: {f['Beam type'][()].decode('utf-8')} != {species}")
                if np.array_equal(f['Grid [m]'][()], grid) == False:
                    raise ValueError(f"Grid mismatch: {f['Grid [m]'][()]} != {grid}")
                energy = f['Beam energy [keV]'][()]
                species=f['Beam type'][()].decode("utf-8")
                grid=f['Grid [m]'][()]
            density=np.vstack([density, f['Density [m-3]'][()]])
            emission=np.vstack([emission, f['Population [-]'][()]])
            temperature=np.vstack([temperature, f['Temperature profiles [eV]'][()]])
    temperature=temperature.mean(axis=0)
    id=bes_obj_id
    verbose=kind + " HESEL synthetic data with average temperature profile"
    tags=[str(i) for i in range(density.shape[0])]
    bes_data=besInferenceDatapoints(grid=grid,energy=energy,species=species,ID=id,temperature=temperature,verbose=verbose)
    bes_data.add_datapoints_bulk(density, emission, tags)
    print("Created besInferenceDatapoints object with ID:", id, "energy:", energy, "species:", species, "temperature shape:", temperature.shape, 
          "grid shape:", grid.shape, "density shape:", density.shape, "emission shape:", emission.shape)
    return bes_data