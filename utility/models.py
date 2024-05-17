import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utility.tfelm.tfelm.ml_elm import ML_ELM

import logging


'''
In this File all the custom model architectures can be created.
'''


class MODELS():
    
    
    def __init__(self, **kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Creating a Logger
        logging.basicConfig(filename=self.logDir + '/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        Creating a Extreme Learning Machine 'https://en.wikipedia.org/wiki/Extreme_learning_machine'
    
        '''
        model = ML_ELM(input_size=self.numPoints, output_size=self.numPoints, name='mlelm1')
        for _ in range(self.numLayers):
            model.add_layer(n_neurons=self.nNodes, activation=self.activationFnc, l2norm=1e2) 
            
        logging.info('ELM model has been created successfully')
        
        return model
    
