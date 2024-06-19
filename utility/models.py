import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from density.inputDict import inputDictionary
import logging

import pyrcn
from pyrcn.extreme_learning_machine import ELMRegressor
from pyrcn.base.blocks import BatchIntrinsicPlasticity
from pyrcn.base.blocks import InputToNode
import sklearn
from sklearn.pipeline import Pipeline


'''
In this File all the custom model architectures can be created.
'''


class MODELS(inputDictionary):
    
    
    def __init__(self):
        
        super().__init__
        
        return
    
    
    def makeCNN(self):
        '''
        Creating a Convolutional Neural Network 'https://en.wikipedia.org/wiki/Convolutional_neural_network'
        
        Returns:
        model (tf.keras.model)
        '''
        inputTensor = Input(shape=(self.numPoints,))  # Define input layer

        g = inputTensor
        for _ in range(self.numLayers):
            g = Dense(self.numNodes, activation=self.activationFnc)(g)
            g = Dropout(self.dropoutFreq)(g)

        outputTensor = Dense(self.numPoints, activation = "linear")(g)  # Output layer

        model = Model(inputs=inputTensor, outputs=outputTensor)

        logging.info('CNN model has been created successfully')

        return model
    
    def makeELM(self):
        '''
        Creating a Extreme Learning Machine (ELM) using the pyrcn package https://pyrcn.readthedocs.io/en/latest/.
        Currently only cascaded ELM networks are supported.
    
        '''

        pipeline_steps = []
        for i in range(self.numLayers):
            if self.layerType[i] == "InputToNode":
                pipeline_steps.append((self.layerName[i], InputToNode(hidden_layer_size = self.numNodes[i], input_activation = self.activationFnc[i], input_scaling = self.inScale[i],
                                                                   input_shift = self.inShift[i], bias_scaling = self.biScale[i], bias_shift = self.biShift[i], random_state = self.random_seed)))
            elif self.layerType[i] == "BatchIntrinsicPlasticity":
                pipeline_steps.append((self.layerName[i], BatchIntrinsicPlasticity(distribution = self.distribution[i], hidden_layer_size = self.numNodes[i], algorithm = self.algorithm[i],
                                                                                input_activation = self.activationFnc[i], random_state = self.random_seed)))
            else:
                logging.ERROR("Unknown layerType: \"" + str(self.layerType) + "\" selected in ELM initialization.")
        
        model = ELMRegressor(input_to_node = Pipeline(steps = pipeline_steps),regressor = self.regressor)
        
        
        logging.info('ELM model has been created successfully')
        
        return model
    
