# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:06:08 2025

@author: asztalos
"""

import h5py as h5
import numpy as np

class besInferenceDatapoints():
    
    def __init__(self, grid, energy, species, ID, zeff=None, q=None, temperature=None, verbose=None, path=None):
        if path is not None:
            self.__load_data_struct(path=path)
        else:
            self.species = species
            self.energy = energy
            self.grid = grid
            self.resolution = len(grid)
            self.ID = ID
            self.zeff = zeff
            self.q = q
            self.temperature = temperature
            self.verbose = verbose
            self.datapoints = []
    
    def __load_data_struct(self, path):
        with h5.File(path, 'r') as h5file:
            self.grid = h5file['grid'][()]
            self.energy = h5file['energy'][()]
            self.species = h5file['species'][()].decode('utf-8')
            self.resolution = h5file['resolution'][()]
            self.zeff = h5file['zeff'][()]
            self.q = h5file['q'][()]
            self.ID = h5file['ID'][()].decode('utf-8')
            self.temperature = h5file['temperature'][()]
            self.verbose = h5file['verbose'][()].decode('utf-8')
            self.datapoints = []
            self.add_datapoints_bulk(densities=h5file['density'][()], 
                                     emissions=h5file['emission'][()],
                                     tags=list(h5file['tags'][()]))
              
    def add_datapoint(self, density, emission, tag):
        if len(density) == self.resolution and len(emission) == self.resolution:
            self.datapoints.append({'density': density, 'emission': emission, 'tag': tag})
        else:
            raise ValueError('The input emission or density arrays do not match the size of the grid. Grid size: ' + 
                             str(self.resolution) + ' Density size: ' + str(len(density)) + ' Emission size: ' + str(len(emission)))
    
    def add_datapoints_bulk(self, densities, emissions, tags):
        if densities.shape[1] != self.resolution or emissions.shape[1] != self.resolution:
            raise ValueError('The resolution of the input 2D density or emission arrays do not match that of the grid. Grid size: ' + str(self.resolution) + 
                             ' Density size: ' + str(densities.shape[1]) + ' Emission size: ' + str(emissions.shape[1]))
        elif densities.shape[0] != len(tags) or emissions.shape[0] != len(tags):
            raise ValueError('There is a mismatch in the number of datapoints. Density: ' + str(densities.shape[0]) +
                             ' Emission: ' + str(emissions.shape[0]) + ' and Tag: ' +str(len(tags)) + ' datapoints.')
        else:
            for tag in range(len(tags)):
                self.datapoints.append({'density': densities[tag, :],
                                        'emission': emissions[tag, :],
                                        'tag': tags[tag]})
    
    def get_datapoints(self):
        densities, emissions = np.array((len(self.datapoints), self.resolution))
        tags = []
        for data_index in range(len(self.datapoints)):
            tags.append(self.datapoints[data_index]['tag'])
            densities[data_index, :] = self.datapoints[data_index]['density']
            emissions[data_index, :] = self.datapoints[data_index]['emission']
        return densities, emissions, tags
    
    def export_to_h5(self, path):
        densities, emissions, tags = self.get_datapoints()
        sdt=h5.string_dtype(encoding='utf-8')
        h5filename = 'Dataset_'+self.species+'_'+str(self.energy)+'_'+self.ID+'.h5'
        with h5.File(path+'/'+h5filename, 'w') as h5file:
            h5file.create_dataset('density', data=densities)
            h5file.create_dataset('emission', data=emissions)
            h5file.create_dataset('grid', data=self.grid)
            h5file.create_dataset('energy', data=self.energy)
            h5file.create_dataset('resolution', data=self.resolution)
            h5file.create_dataset('tags', (len(self.datapoints),), dtype=sdt, data=tags)
            h5file.create_dataset('species', dtype=sdt, data=self.species)
            h5file.create_dataset('ID', dtype=sdt, data=self.ID)
            h5file.create_dataset('Zeff', data=self.zeff)
            h5file.create_dataset('q', data=self.q)
            h5file.create_dataset('temperature', data=self.temperature)
            h5file.create_dataset('verbose', dtype=sdt, data=self.verbose)
