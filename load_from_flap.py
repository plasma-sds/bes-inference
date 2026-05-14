import os
import flap
import numpy as np

from neuro_bes.data import besInferenceDatapoints

def w7x_experimental_flap2database(path, shot, save_path="/home/molnarbalazs/data/BES_ML_modelling"):
    """
    Loads W7X experimental data from flap files.
    Curates it for good reconstructions.
    Loads data to BES dataobject.
    Exports to HDF5 format.
    Only keeps data where density reconstruction is available and where the 
    reconstructed light profile is close to the measured light profile (good reconstructions).

    Parameters
    ----------
    path : str
        Path to the directory containing the shot data.
    shot : str
        Shot number.
    save_path : str, optional
        Path to the directory where the HDF5 file will be saved.
        
    Returns
    -------
    None.
    """
    # get filenames from shot directory
    try:
        file_list=os.listdir(os.path.join(path,shot))
    except FileNotFoundError:
        raise ValueError("Directory " + file_list + " does not exist")

    # find light, density, reconstructed light datasources
    # load them with flap
    file_name_light=file_list[[i for i in range(len(file_list)) if ('light_ds_orig' in file_list[i])][0]]
    file_name_density=file_list[[i for i in range(len(file_list)) if ('dens' in file_list[i])][0]]
    file_name_light_recon=file_list[[i for i in range(len(file_list)) if ('light_recon' in file_list[i])][0]]
    light=flap.load(os.path.join(path,shot,file_name_light))
    density=flap.load(os.path.join(path,shot,file_name_density))
    light_recon=flap.load(os.path.join(path,shot,file_name_light_recon))

    # check if density and reconstructed light time instances match
    # Catch when time instances differ (they should match in principle)
    time_instances_light_recon=light_recon.coordinate('Time')[0][:,0]
    time_instances_density=density.coordinate('Time')[0][:,0]
    if not np.any(time_instances_light_recon==time_instances_density):
        raise ValueError("Error: Light recon and density time instances do not match for shot " + shot)

    # keep only time instances where density recon is available
    # discard light profiles where recon is unavailable
    time_instances_light=light.coordinate('Time')[0][:,0]
    mapping = {val: idx for idx, val in enumerate(time_instances_light)}
    mask_timeinstance = np.array([mapping[val] for val in time_instances_density])
    light_data=light.data[mask_timeinstance,:]
    density_data=density.data
    light_recon_data=light_recon.data

    # check if spatial grid is the same for all profiles
    # discard profiles with different grid than the first time instance
    r_coord=light.coordinate('Device R')[0][mask_timeinstance[0]]
    mask_samegrid_1=np.all(light.coordinate('Device R')[0]==r_coord,axis=1)[mask_timeinstance]
    mask_samegrid_2=np.all(density.coordinate('Device R')[0]==r_coord,axis=1)
    light_data=light_data[mask_samegrid_1*mask_samegrid_2]
    density_data=density_data[mask_samegrid_1*mask_samegrid_2]
    light_recon_data=light_recon_data[mask_samegrid_1*mask_samegrid_2]

    # keep only good reconstructions
    # reconstructed light profile must be close to measured light profile 
    mask_goodrecon=np.sqrt(np.mean((light_data-light_recon_data)**2,axis=1))/np.mean(light_data,axis=1)<0.1
    light_data=light_data[mask_goodrecon]
    density_data=density_data[mask_goodrecon]
    light_recon_data=light_recon_data[mask_goodrecon]
    time_instances_density=time_instances_density[mask_goodrecon]

    # prepare BES dataobject 
    # add beam meta
    grid=r_coord
    energy=0
    species="Na"
    ID="we_"+shot
    zeff=0
    q=0
    temperature=np.array(0)
    verbose="W7X experimental data shot " + shot + ", 10Hz averaging, curated for good SPADE recons"

    # create tags with timing information
    tags=['Time instance ' + str(i) + ' s' for i in time_instances_density]

    # create and populate BES dataobject instance
    test_data=besInferenceDatapoints(grid=grid,energy=energy,species=species,ID=ID,zeff=zeff,q=q,temperature=temperature,verbose=verbose)
    test_data.add_datapoints_bulk(density_data, light_data, tags)

    # export to HDF5
    test_data.export_to_h5(path_to_dir=save_path)