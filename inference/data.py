# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:06:08 2025

@author: asztalos
"""

import numpy as np

class besInferenceDatapoints():
    
    def __init__(self, grid, energy, species, zeff, ID, temperature, q, verbose=None, path=None):
        if path is not None:
            self.__load_data_struct(path=path)
        

        self.species = species
        self.energy = energy
        self.grid = grid
        self.resolution = len(grid)
        self.zeff = zeff
        self.q = q
        self.temperature = temperature
        self.ID = ID
        self.verbose = verbose
        self.datapoints = []
    
    def __load_data_struct(self, path):
        pass
    
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
            raise ValueError('There is a mismatch in the number of Density: ' + str(densities.shape[0]) +
                             ' Emission: ' + str(emissions.shape[0]) + ' and Tag: ' +str(len(tags)) + ' datapoints.')
        else:
            for tag in range(len(tags)):
                self.datapoints.append({'density': densities[tag, :],
                                        'emission': emissions[tag, :],
                                        'tag': tags[tag]})
    
    def get_datapoints(self):
        density, emission = np.array(())
    
    def export_to_h5(self):
        pass
        