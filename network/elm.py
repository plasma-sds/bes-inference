# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:35:38 2024

@author: Örs
"""
from typing import Any
import os
import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import logging
import mlflow
# Append the path of the utility folder to sys.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utility.lossFunctions import LOSSES
from utility.models import MODELS
from utility.dataLoader import DATA_LOADER
from density.inputDict import inputDictionary

#importing ELM specific packages
import pyrcn
from pyrcn.extreme_learning_machine import ELMRegressor
from pyrcn.base.blocks import BatchIntrinsicPlasticity
from pyrcn.base.blocks import InputToNode
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, FeatureUnion
from pyrcn.base.blocks import InputToNode

class ELMTRAINER(inputDictionary):


    def __init__(self):

        # Creating the timestring/ runname
        super().__init__()
        
        dic = {key:value for key, value in inputDictionary.__dict__.items() if not key.startswith('__') and not callable(key)}
        mlflow.log_params(dic)
   
        logging.info('Input Json File Saved')
            
        return




    def evaluate(self, inputData: tf.Tensor, outputData : tf.Tensor, train_eval : bool):
        """
        Args:
            inputData (tf.Tensor) [None, self.numPoints]: The input data for the ELM 
            outputData (tf.Tensor) [None, self.numPoints]: The known data for the corresponding inputData
            
        Note: in and output data will be converted to np.array, due to pyrcn's incompatibility with tf.Tensor objects

        Returns:
            losses (tf.Tensors): All the individual losses obtained by the different loss functions
            total_loss (tf.Tensors)    : One weighted loss 
        """
        
        if train_eval == True:
            self.model.fit(inputData.numpy(), outputData.numpy())
            
        predictions = self.model.predict(inputData.numpy())
        
        total_loss = 0
        losses = []
        for loss_fn, loss_weight in zip(self.losses, self.loss_weights):
            loss = loss_fn(outputData, predictions)
            losses.append(loss)
            total_loss += loss_weight * loss

        return losses, total_loss
    
    def get_model(self) -> None:
        '''
        Fetching the model that should be used for training
        '''
        if self.model_name == 'cnn':
            self.model = MODELS().makeCNN()
        elif self.model_name == 'elm':
            self.model = MODELS().makeELM()
        else:
            logging.ERROR('The model_name you selected is not available.')
        
        return 
    
    def get_data(self) -> None:
        '''
        Fetching the data used for training and validation and testing.
        '''
        
        loader = DATA_LOADER()
        
        self.sensorWeights = loader.return_sensor_weights
        self.InputTrain, self.InputTest, self.InputVal, self.OutputTrain, self.OutputTest, self.OutputVal = loader.fetch_data()
        
        return


    def compile_model(self) -> None:
        '''
        Fetching the Loss functions from the lossFunction class
        Initializing the Learning Rate
        Initializing the Optimizer
        '''

        logging.info('The Model has Compiled with metrics')
        
        return

    def log_metrics(self, losses : tf.Tensor, train_eval : bool) -> None:
        
        """
        Logs the metrics into mlflow
        """
        

        if train_eval:
            for loss, name, weight in zip(losses, self.losses_str, self.loss_weights):
                if weight != 0:
                    mlflow.log_metric("loss_" + name, loss.numpy())
                else:
                    mlflow.log_metric("metric_" + name, loss.numpy())
        else:
            for loss, name in zip(losses, self.losses_str):
                mlflow.log_metric('val_' + name, loss.numpy())
                
        return
    
    
    def train_model(self) -> None:

        train_losses, total_train_loss = self.evaluate(inputData=self.InputTrain, outputData=self.OutputTrain, train_eval = True)
        val_losses, total_val_loss     = self.evaluate(inputData=self.InputVal, outputData=self.OutputVal, train_eval = False)
            
        self.log_metrics(losses = train_losses, train_eval = True)
        self.log_metrics(losses = val_losses, train_eval = False)
                 
        return 

    def final_save(self) -> None:

        # Define the files to handle
        files = ["InputTest.npy", "OutputTest.npy", "InputTrain.npy", "OutputTrain.npy", "InputVal.npy", "OutputVal.npy", "sensorWeights.npy"]

        # Saving, logging, and removing files
        for file in files:

            np.save(os.path.join(self.logDir, file), getattr(self, file.split('.')[0])) # Save the file
            mlflow.log_artifact(os.path.join(self.logDir, file))# Log the file
            os.remove(os.path.join(self.logDir, file))# Remove the file
            
        files = ["log.txt"]
        
        # Saving, logging, and removing files
        for file in files:

            mlflow.log_artifact(file)# Log the file
            os.remove(file)# Remove the file
        
        return
