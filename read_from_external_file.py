import numpy as np
import h5py
import pandas as pd
import os

def read_synthetic_static_data(file_path):
    df = pd.read_hdf(file_path,"df")
    df = df[df['Density Shape[$1/m^3$]'].apply(lambda x: hasattr(x, "__len__"))]
    density = np.array([x[0:400] for x in df['Density Shape[$1/m^3$]']])
    emission = np.array([x[0:400] for x in df['Emission 3p-->3s']])
    x=np.linspace(0, 1, 400)
    return x, emission, density

def read_synthetic_fluctuation_data(path_to_directory):
    files = [os.path.join(path_to_directory, f) for f in os.listdir(path_to_directory) if f.endswith(('.h5', '.hdf5'))]
    df_list = [pd.read_hdf(f, "df") for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df = df[df['Density Shape[$1/m^3$]'].apply(lambda x: hasattr(x, "__len__"))]
    density = np.array([x[0:400] for x in df['Density Shape[$1/m^3$]']])
    emission = np.array([x[0:400] for x in df['Emission 3p-->3s']])
    x=np.linspace(0, 1, 400)
    return x, emission, density

def read_asdex_experimental(file_path):
    with h5py.File(file_path, 'r') as f:
        x = f['s40701']["x2"][()]
        emission=np.mean(f['s40701']["lib2dat"][()],axis=1)
        density=f['s40701']["ne"][()]
    return x, emission, density    

def read_w7x_experimental(file_path):
    with h5py.File(file_path, 'r') as f:
        emission = f["/emission"][()]
        density = f["/density"][()]
        r_coord = f["/Device R"][()]
    return r_coord, emission, density

def read_hesel(path_to_directory,kind='fast'):
    emission=[]
    density=[]
    r_coord=[]
    file_paths=[]
    for filename in os.listdir(path_to_directory):
        if filename.endswith('.h5') and kind=='fast':
            if 'fast' in filename:
                file_paths.append(os.path.join(path_to_directory, filename))
        if filename.endswith('.h5') and kind=='slow':
            if 'fast' not in filename:
                file_paths.append(os.path.join(path_to_directory, filename))
    for file_path in file_paths:
        print(file_path)
        with h5py.File(file_path, 'r') as f:
            density.append(np.asarray(f["Density [m-3]"][:]))
            emission.append(np.asarray(f["Population [-]"][:]))
            r_coord.append(np.asarray(f["Grid [m]"][:]))
    try:
        density = np.concatenate(density,axis=0)
        emission = np.concatenate(emission,axis=0)
        r_coord = np.stack(r_coord)
    except ValueError:
        print("Datasets have incompatible shapes. Keeping data as list.")
    return r_coord, emission, density