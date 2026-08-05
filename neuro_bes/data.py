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
            self.density_errors = np.empty((0, self.resolution))
            self.emission_errors = np.empty((0, self.resolution))
            self.tags = []

    def __load_data_struct(self, path):
        with h5.File(path, 'r') as h5file:
            self.grid = h5file['grid'][()]
            energy_val = float(h5file['energy'][()])
            self.energy = int(energy_val) if not np.isnan(energy_val) else np.nan
            self.species = h5file['species'][()].decode('utf-8')
            self.resolution = int(h5file['resolution'][()])
            self.zeff = float(h5file['zeff'][()])
            self.q = float(h5file['q'][()])
            self.ID = h5file['ID'][()].decode('utf-8')
            self.temperature = h5file['temperature'][()]
            self.verbose = h5file['verbose'][()].decode('utf-8')
            self.densities = np.empty((0, self.resolution))
            self.emissions = np.empty((0, self.resolution))
            self.density_errors = np.empty((0, self.resolution))
            self.emission_errors = np.empty((0, self.resolution))
            self.tags = []
            density_errors = h5file['density_error'][()] if 'density_error' in h5file else None
            emission_errors = h5file['emission_error'][()] if 'emission_error' in h5file else None
            self.add_datapoints_bulk(densities=h5file['density'][()],
                                     emissions=h5file['emission'][()],
                                     tags=list([s.decode('utf-8') for s in h5file['tags'][()]]),
                                     density_errors=density_errors,
                                     emission_errors=emission_errors)
              
    def add_datapoint(self, density, emission, tag, density_error=None, emission_error=None):
        """
        The function adds individual datapoints of density-emission pairs with tag descriptor to a list of datapoints.

        Parameters
        ----------
        density :        TYPE 1D numpy array     DESCRIPTION 1D array of the plasma density profile along the beam in [m-3]
        emission :       TYPE 1D numpy array     DESCRIPTION 1D array of the emission profile along the beam in [a.u.]
        tag :            TYPE string             DESCRIPTION is specific descriptor for the origin of the datapoint
        density_error :  TYPE 1D numpy array     DESCRIPTION optional 1D array of the density uncertainty in [m-3], same size as ``density``. If not provided, an all-NaN row is stored.
        emission_error : TYPE 1D numpy array     DESCRIPTION optional 1D array of the emission uncertainty in [a.u.], same size as ``emission``. If not provided, an all-NaN row is stored.

        Raises
        ------
        ValueError
            DESCRIPTION is raised if the spatial resolution of the density, emission, or
            (when provided) their error arrays do not match that of the grid.

        Returns
        -------
        None.

        """
        density = np.asarray(density)
        emission = np.asarray(emission)
        if len(density) != self.resolution or len(emission) != self.resolution:
            raise ValueError('The input emission or density arrays do not match the size of the grid. Grid size: ' +
                             str(self.resolution) + ' Density size: ' + str(len(density)) + ' Emission size: ' + str(len(emission)))

        if density_error is None:
            density_error_row = np.full(self.resolution, np.nan)
        else:
            density_error_row = np.asarray(density_error)
            if len(density_error_row) != self.resolution:
                raise ValueError('The input density_error array does not match the size of the grid. Grid size: ' +
                                 str(self.resolution) + ' density_error size: ' + str(len(density_error_row)))

        if emission_error is None:
            emission_error_row = np.full(self.resolution, np.nan)
        else:
            emission_error_row = np.asarray(emission_error)
            if len(emission_error_row) != self.resolution:
                raise ValueError('The input emission_error array does not match the size of the grid. Grid size: ' +
                                 str(self.resolution) + ' emission_error size: ' + str(len(emission_error_row)))

        self.densities = np.concatenate([self.densities, density[np.newaxis, :]], axis=0)
        self.emissions = np.concatenate([self.emissions, emission[np.newaxis, :]], axis=0)
        self.density_errors = np.concatenate([self.density_errors, density_error_row[np.newaxis, :]], axis=0)
        self.emission_errors = np.concatenate([self.emission_errors, emission_error_row[np.newaxis, :]], axis=0)
        self.tags.append(tag)
    
    def add_datapoints_bulk(self, densities, emissions, tags, density_errors=None, emission_errors=None):
        """
        The function adds density-emission-tag datapoints in bulk to the object.

        Parameters
        ----------
        densities :       TYPE 2D numpy array      DESCRIPTION 2D array of the plasma density profiles along the beam in [m-3]
        emissions :       TYPE 2D numpy array      DESCRIPTION 2D array of the emission profiles along the beam in [a.u.]
        tags :            TYPE 1D list             DESCRIPTION 1D list of descriptors for each datapoint.
        density_errors :  TYPE 2D numpy array      DESCRIPTION optional 2D array of density uncertainties in [m-3], same shape as ``densities``. If not provided, an all-NaN block of matching shape is stored.
        emission_errors : TYPE 2D numpy array      DESCRIPTION optional 2D array of emission uncertainties in [a.u.], same shape as ``emissions``. If not provided, an all-NaN block of matching shape is stored.

        Raises
        ------
        ValueError
            DESCRIPTION is raised if the spatial resolution of the density, emission, or
            (when provided) their error arrays do not match that of the grid, OR the number
            of density, emission, tag (and, when provided, error) datapoints do not match.

        Returns
        -------
        None.

        """
        densities = np.asarray(densities)
        emissions = np.asarray(emissions)
        if densities.shape[1] != self.resolution or emissions.shape[1] != self.resolution:
            raise ValueError('The resolution of the input 2D density or emission arrays do not match that of the grid. Grid size: ' + str(self.resolution) +
                             ' Density size: ' + str(densities.shape[1]) + ' Emission size: ' + str(emissions.shape[1]))
        if densities.shape[0] != len(tags) or emissions.shape[0] != len(tags):
            raise ValueError('There is a mismatch in the number of datapoints. Density: ' + str(densities.shape[0]) +
                             ' Emission: ' + str(emissions.shape[0]) + ' and Tag: ' + str(len(tags)) + ' datapoints.')

        n_new = densities.shape[0]

        if density_errors is None:
            density_errors_arr = np.full((n_new, self.resolution), np.nan)
        else:
            density_errors_arr = np.asarray(density_errors)
            if density_errors_arr.shape != (n_new, self.resolution):
                raise ValueError('The shape of density_errors ' + str(density_errors_arr.shape) +
                                 ' does not match the expected shape ' + str((n_new, self.resolution)) + '.')

        if emission_errors is None:
            emission_errors_arr = np.full((n_new, self.resolution), np.nan)
        else:
            emission_errors_arr = np.asarray(emission_errors)
            if emission_errors_arr.shape != (n_new, self.resolution):
                raise ValueError('The shape of emission_errors ' + str(emission_errors_arr.shape) +
                                 ' does not match the expected shape ' + str((n_new, self.resolution)) + '.')

        self.densities = np.concatenate([self.densities, densities], axis=0)
        self.emissions = np.concatenate([self.emissions, emissions], axis=0)
        self.density_errors = np.concatenate([self.density_errors, density_errors_arr], axis=0)
        self.emission_errors = np.concatenate([self.emission_errors, emission_errors_arr], axis=0)
        self.tags.extend(list(tags))
    
    def get_datapoints(self, include_errors=False):
        """
        The function returns 2D arrays for other processing of the density-emission and corresponding tags.

        Parameters
        ----------
        include_errors : TYPE bool           DESCRIPTION if True, the density and emission uncertainty arrays are returned alongside the datapoints. Default is False, preserving the legacy 3-tuple return.

        Returns
        -------
        When ``include_errors`` is False (default):
            densities :       TYPE 2D numpy array      DESCRIPTION 2D array of the plasma density profiles along the beam in [m-3]
            emissions :       TYPE 2D numpy array      DESCRIPTION 2D array of the emission profiles along the beam in [a.u.]
            tags :            TYPE 1D list             DESCRIPTION 1D list of descriptors for each datapoint.
        When ``include_errors`` is True:
            densities :       TYPE 2D numpy array      DESCRIPTION 2D array of the plasma density profiles along the beam in [m-3]
            emissions :       TYPE 2D numpy array      DESCRIPTION 2D array of the emission profiles along the beam in [a.u.]
            density_errors :  TYPE 2D numpy array      DESCRIPTION 2D array of density uncertainties in [m-3] (NaN where not provided)
            emission_errors : TYPE 2D numpy array      DESCRIPTION 2D array of emission uncertainties in [a.u.] (NaN where not provided)
            tags :            TYPE 1D list             DESCRIPTION 1D list of descriptors for each datapoint.

        Note
        ----
        The returned density, emission, and error arrays are references to the
        internal storage (no copy is made). Use ``np.copy`` if the caller
        intends to mutate them. The tags list is a fresh copy.
        """
        if include_errors:
            return self.densities, self.emissions, self.density_errors, self.emission_errors, list(self.tags)
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
            h5file.create_dataset('density_error', data=np.asarray(self.density_errors, dtype=np.float64))
            h5file.create_dataset('emission_error', data=np.asarray(self.emission_errors, dtype=np.float64))
            h5file.create_dataset('grid', data=self.grid)
            h5file.create_dataset('energy', data=np.float64(self.energy))
            h5file.create_dataset('resolution', data=self.resolution)
            h5file.create_dataset('tags', (len(self.tags),), dtype=sdt, data=self.tags)
            h5file.create_dataset('species', dtype=sdt, data=self.species)
            h5file.create_dataset('ID', dtype=sdt, data=self.ID)
            h5file.create_dataset('zeff', data=np.float64(self.zeff))
            h5file.create_dataset('q', data=np.float64(self.q))
            h5file.create_dataset('temperature', data=np.asarray(self.temperature, dtype=np.float64))
            h5file.create_dataset('verbose', dtype=sdt, data=self.verbose)
        print('File saved to h5: ' + os.path.join(path_to_dir, h5filename))
