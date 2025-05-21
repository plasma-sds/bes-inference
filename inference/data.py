# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:06:08 2025

@author: asztalos
"""

class besInferenceDatapoints():
    
    def __init__(self, grid, energy, species, zeff, temperature, q, ID, verbose=None, path=None):
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
            raise ValueError('The input emission or density arrays do not match the size of the grid.')
    
    def add_datapoints_bulk(self, densities, emissions, tags):
        pass
    
    def get_datapoints(self):
        pass
    
    def export_to_h5(self):
        pass
        