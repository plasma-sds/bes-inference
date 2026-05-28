# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:06:08 2025

@author: asztalos
"""

import os
import h5py as h5
import numpy as np

class besInferenceDatapoints():
    
    def __init__(self, grid=np.array, energy=None, species=str, ID=str, zeff=None,
                 q=None, temperature=None, verbose=str, path=None):
        """
        The object is desigend to hold datapoints for BES inference

        Parameters
        ----------
        grid :          TYPE, 1D numpy array,   DESCRIPTION 1D spatial grid where density-emission pairs are featured in [m]. The default is np.array.
        energy :        TYPE, integer,          DESCRIPTION feature the beam energy in [keV]. If not provided, defaults to np.nan.
        species :       TYPE, string,           DESCRIPTION features the type of beam material ex: H, Li, Na. The default is str.
        ID :            TYPE, string            DESCRIPTION shows the ID for the datapoints. The default is str.
        zeff :          TYPE, float             DESCRIPTION is the zeff of the plasma profile. If not provided, defaults to np.nan.
        q :             TYPE, float             DESCRIPTION is the average impurity charge in the plasma. If not provided, defaults to np.nan.
        temperature :   TYPE, 2D numpy array    DESCRIPTION 2D array of the plasma temperature profiles in [eV]. If not provided, defaults to np.empty((0, resolution)) filled with np.nan.
        verbose :       TYPE, string            DESCRIPTION contains a more verbose desciption of the dataset, beyond the ID. The default is str.
        path :          TYPE, string            DESCRIPTION gives the path to a h5 file to load the data for inference. The default is None.

        Returns
        -------
        None.

        """
        if path is not None:
            self.__load_data_struct(path=path)
        else:
            self.species = species
            self.energy = energy if energy is not None else np.nan
            self.grid = grid
            self.resolution = len(grid)
            self.ID = ID
            self.zeff = zeff if zeff is not None else np.nan
            self.q = q if q is not None else np.nan
            self.temperature = temperature if temperature is not None else np.empty((0, self.resolution)) * np.nan
            self.verbose = verbose
            self.densities = np.empty((0, self.resolution))
            self.emissions = np.empty((0, self.resolution))
            self.tags = []

    def __load_data_struct(self, path):
        with h5.File(path, 'r') as h5file:
            self.grid = h5file['grid'][()]
            self.energy = int(h5file['energy'][()])
            self.species = h5file['species'][()].decode('utf-8')
            self.resolution = int(h5file['resolution'][()])
            self.zeff = float(h5file['zeff'][()])
            self.q = float(h5file['q'][()])
            self.ID = h5file['ID'][()].decode('utf-8')
            self.temperature = h5file['temperature'][()]
            self.verbose = h5file['verbose'][()].decode('utf-8')
            self.densities = np.empty((0, self.resolution))
            self.emissions = np.empty((0, self.resolution))
            self.tags = []
            self.add_datapoints_bulk(densities=h5file['density'][()], 
                                     emissions=h5file['emission'][()],
                                     tags=list([s.decode('utf-8') for s in h5file['tags'][()]]))
              
    def add_datapoint(self, density, emission, tag):
        """
        The function adds individual datapoints of density-emission pairs with tag descriptor to a list of datapoints.

        Parameters
        ----------
        density :   TYPE 1D numpy array     DESCRIPTION 1D array of the plasma density profile along the beam in [m-3]
        emission :  TYPE 1D numpy array     DESCRIPTION 1D array of the emission profile along the beam in [a.u.]
        tag :       TYPE string             DESCRIPTION is specific descriptor for the origin of the datapoint

        Raises
        ------
        ValueError
            DESCRIPTION is raised if the spatial resolution of the density or emission arrays do not match that of the grid. 

        Returns
        -------
        None.

        """
        density = np.asarray(density)
        emission = np.asarray(emission)
        if len(density) == self.resolution and len(emission) == self.resolution:
            self.densities = np.concatenate([self.densities, density[np.newaxis, :]], axis=0)
            self.emissions = np.concatenate([self.emissions, emission[np.newaxis, :]], axis=0)
            self.tags.append(tag)
        else:
            raise ValueError('The input emission or density arrays do not match the size of the grid. Grid size: ' + 
                             str(self.resolution) + ' Density size: ' + str(len(density)) + ' Emission size: ' + str(len(emission)))
    
    def add_datapoints_bulk(self, densities, emissions, tags):
        """
        The function adds density-emission-tag datapoints in bulk to the object.

        Parameters
        ----------
        densities : TYPE 2D numpy array      DESCRIPTION 2D array of the plasma density profiles along the beam in [m-3]
        emissions : TYPE 2D numpy array      DESCRIPTION 2D array of the emission profiles along the beam in [a.u.]
        tags :      TYPE 1D list             DESCRIPTION 1D list of descriptors for each datapoint.

        Raises
        ------
        ValueError
            DESCRIPTION is raised if the spatial resolution of the density or emission arrays do not match that of the grid 
            OR the number of density, emission, tag datapoints do not match.

        Returns
        -------
        None.

        """
        densities = np.asarray(densities)
        emissions = np.asarray(emissions)
        if densities.shape[1] != self.resolution or emissions.shape[1] != self.resolution:
            raise ValueError('The resolution of the input 2D density or emission arrays do not match that of the grid. Grid size: ' + str(self.resolution) + 
                             ' Density size: ' + str(densities.shape[1]) + ' Emission size: ' + str(emissions.shape[1]))
        elif densities.shape[0] != len(tags) or emissions.shape[0] != len(tags):
            raise ValueError('There is a mismatch in the number of datapoints. Density: ' + str(densities.shape[0]) +
                             ' Emission: ' + str(emissions.shape[0]) + ' and Tag: ' +str(len(tags)) + ' datapoints.')
        else:
            self.densities = np.concatenate([self.densities, densities], axis=0)
            self.emissions = np.concatenate([self.emissions, emissions], axis=0)
            self.tags.extend(list(tags))
    
    def get_datapoints(self):
        """
        The function returns 2D arrays for other processing of the density-emission and corresponding tags.

        Returns
        -------
        densities : TYPE 2D numpy array      DESCRIPTION 2D array of the plasma density profiles along the beam in [m-3]
        emissions : TYPE 2D numpy array      DESCRIPTION 2D array of the emission profiles along the beam in [a.u.]
        tags :      TYPE 1D list             DESCRIPTION 1D list of descriptors for each datapoint.

        Note
        ----
        The returned density and emission arrays are references to the
        internal storage (no copy is made). Use ``np.copy`` if the caller
        intends to mutate them. The tags list is a fresh copy.
        """
        return self.densities, self.emissions, list(self.tags)
    
    def export_to_h5(self, path_to_dir):
        """
        The functions stores the data objects content into H5 format.

        Parameters
        ----------
        path_to_dir : TYPE string           DESCRIPTION location of the folder where the file is to be saved. The code generates the name of the file.

        Returns
        -------
        None.

        """
        sdt = h5.string_dtype(encoding='utf-8')
        h5filename = 'Dataset_' + self.species + '_' + str(self.energy) + '_' + self.ID + '.h5'
        with h5.File(os.path.join(path_to_dir, h5filename), 'w') as h5file:
            h5file.create_dataset('density', data=self.densities)
            h5file.create_dataset('emission', data=self.emissions)
            h5file.create_dataset('grid', data=self.grid)
            h5file.create_dataset('energy', data=self.energy)
            h5file.create_dataset('resolution', data=self.resolution)
            h5file.create_dataset('tags', (len(self.tags),), dtype=sdt, data=self.tags)
            h5file.create_dataset('species', dtype=sdt, data=self.species)
            h5file.create_dataset('ID', dtype=sdt, data=self.ID)
            h5file.create_dataset('zeff', data=self.zeff)
            h5file.create_dataset('q', data=self.q)
            h5file.create_dataset('temperature', data=self.temperature)
            h5file.create_dataset('verbose', dtype=sdt, data=self.verbose)
        print('File saved to h5: ' + os.path.join(path_to_dir, h5filename))
