import os
import flap
import numpy as np

from neuro_bes.data import besInferenceDatapoints

def w7x_experimental_flap2database(path, shot, averaging=False, curation=True, save_path="/home/molnarbalazs/data/BES_ML_modelling"):
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
    averaging : bool, optional
        Whether to read averaged data.
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
        raise ValueError("Directory " + os.path.join(path,shot) + " does not exist")

    # find light, density, reconstructed light datasources
    # load them with flap

    file_name_light_ds=[i for i in file_list if ('light_ds_orig' in i)]
    file_name_density=[i for i in file_list if ('dens' in i)]
    file_name_light_recon=[i for i in file_list if ('light_recon' in i)]
    if len(file_name_density)>1:
        raise ValueError("There is more than one data source for shot " + shot)
    if len(file_name_density)==0:
        raise ValueError("Data source not found for shot " + shot)

    file_name_light_ds=file_name_light_ds[0]
    file_name_density=file_name_density[0]
    file_name_light_recon=file_name_light_recon[0]
    if not averaging:
        file_name_light=[i for i in file_list if ('light_orig' in i)]
        file_name_light=file_name_light[0]
        # if the file named file_name_light is much larger than file_name_light_ds, than fast modulation was used, otherwise slow modulation
        fast_modulation=os.path.getsize(os.path.join(path,shot,file_name_light))>10*os.path.getsize(os.path.join(path,shot,file_name_light_ds))
        light=flap.load(os.path.join(path,shot,file_name_light))
    light_ds=flap.load(os.path.join(path,shot,file_name_light_ds))
    density=flap.load(os.path.join(path,shot,file_name_density))
    light_recon=flap.load(os.path.join(path,shot,file_name_light_recon))

    # check if density and reconstructed light time instances match
    # Catch when time instances differ (they should match in principle)
    time_instances_light_recon=light_recon.coordinate('Time')[0][:,0]
    time_instances_density=density.coordinate('Time')[0][:,0]
    avg_timedelta=np.mean(np.diff(time_instances_density))
    if avg_timedelta<0.01:
        raise ValueError(f"Downsampling frequency is above limit ({int(1/avg_timedelta)} Hz) in shot " + shot)
    if not np.any(time_instances_light_recon==time_instances_density):
        raise ValueError("Error: Light recon and density time instances do not match for shot " + shot)

    # keep only time instances where density recon is available
    # discard light profiles where recon is unavailable
    if not averaging:
        time_instances_light=light.coordinate('Time')[0][:,0]
        mapping = {val: idx for idx, val in enumerate(time_instances_light)}
        mask_timeinstance = np.array([mapping[val] for val in time_instances_density])
        light_data=light.data[mask_timeinstance,:]
    time_instances_light=light_ds.coordinate('Time')[0][:,0]
    mapping = {val: idx for idx, val in enumerate(time_instances_light)}
    mask_timeinstance = np.array([mapping[val] for val in time_instances_density])
    light_data_ds=light_ds.data[mask_timeinstance,:]

    density_data=density.data
    light_recon_data=light_recon.data

    # check if spatial grid is the same for all profiles
    # discard profiles with different grid than the first time instance
    r_coord=light_ds.coordinate('Device R')[0][mask_timeinstance[0]]
    mask_samegrid_1=np.all(light_ds.coordinate('Device R')[0]==r_coord,axis=1)[mask_timeinstance]
    mask_samegrid_2=np.all(density.coordinate('Device R')[0]==r_coord,axis=1)
    if not averaging:
        light_data=light_data[mask_samegrid_1*mask_samegrid_2]
    light_data_ds=light_data_ds[mask_samegrid_1*mask_samegrid_2]
    density_data=density_data[mask_samegrid_1*mask_samegrid_2]
    light_recon_data=light_recon_data[mask_samegrid_1*mask_samegrid_2]

    # keep only good reconstructions
    # reconstructed light profile must be close to measured light profile 
    #mask_goodrecon=np.sqrt(np.mean((light_data_ds-light_recon_data)**2,axis=1))/np.mean(light_data_ds,axis=1)<0.1
    if curation:
        mask = light_data_ds > np.max(light_data_ds,axis=1, keepdims=True) * 0.1
        rel_err = np.abs((light_data_ds - light_recon_data) / light_data_ds)
        rel_err = np.where(mask, rel_err, np.nan)
        mask_goodrecon = np.nanmax(rel_err, axis=1) < 0.05
        if not averaging:
            light_data=light_data[mask_goodrecon]
        light_data_ds=light_data_ds[mask_goodrecon]
        density_data=density_data[mask_goodrecon]
        light_recon_data=light_recon_data[mask_goodrecon]
        time_instances_density=time_instances_density[mask_goodrecon]

    # prepare BES dataobject 
    # add beam meta
    grid=r_coord
    energy=0
    species="Na"
    zeff=0
    q=0
    temperature=np.array(0)

    # create tags with timing information
    tags=['Time instance ' + str(i) + ' s' for i in time_instances_density]

    # create and populate experimental BES dataobject instance
    if averaging:
        ID="we_avg_"+shot
        verbose=f"W7X experimental data shot " + shot + ", {int(1/avg_timedelta)}Hz averaging, curated for good SPADE recons"
        test_data=besInferenceDatapoints(grid=grid,energy=energy,species=species,ID=ID,zeff=zeff,q=q,temperature=temperature,verbose=verbose)
        test_data.add_datapoints_bulk(density_data, light_data_ds, tags)
    else:
        ID="we_"+shot
        if fast_modulation:
            verbose=f"W7X experimental data shot " + shot + ", fast modulation, curated for good SPADE recons"
        else:
            verbose=f"W7X experimental data shot " + shot + ", slow modulation, curated for good SPADE recons"
        test_data=besInferenceDatapoints(grid=grid,energy=energy,species=species,ID=ID,zeff=zeff,q=q,temperature=temperature,verbose=verbose)
        test_data.add_datapoints_bulk(density_data, light_data, tags)
    
    # create and populate synthetic BES dataobject instance
    ID="ws_"+shot
    verbose=f"W7X synthetic data shot " + shot + ", curated for good SPADE recons"
    test_data_synthetic=besInferenceDatapoints(grid=grid,energy=energy,species=species,ID=ID,zeff=zeff,q=q,temperature=temperature,verbose=verbose)
    test_data_synthetic.add_datapoints_bulk(density_data, light_recon_data, tags)

    # export to HDF5
    test_data.export_to_h5(path_to_dir=save_path)
    test_data_synthetic.export_to_h5(path_to_dir=save_path)
    print("modulation: " + ("fast" if fast_modulation else "slow"))