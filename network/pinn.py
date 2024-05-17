# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:34:30 2024

@author: Örs
"""
import tensorflow as tf
from typing import Any
import os
import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import logging
import mlflow

from lossFunctions import LOSSES
from models import MODELS
from dataLoader import DATA_LOADER


class TRAINER(LOSSES, MODELS, DATA_LOADER):


    def __init__(self, **kwargs : Any):

        # Creating the timestring/ runname
        
        self.kwargs = kwargs
        self.epoch = 0
        self.best_val_loss = 10**10

        # Writing all the dictionary values to the class
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.DataBatchRatio = int(self.nData / self.batchSize) 
        
        mlflow.log_params(kwargs)
        logging.info('Input Json File Saved')
            
        return



    @tf.function
    def evaluation_step(self, inputData: tf.Tensor, outputData : tf.Tensor, train_eval : bool):
        """
        This performs one forward step through the neural network and if train_eval : True it also applies the gradient to the model

        Args:
            inputData (tf.Tensor) [None, self.numPoints]: The input data for the neural network 
            outputData (tf.Tensor) [None, self.numPoints]: The known data for the corresponding inputData

        Returns:
            losses (list of tf.Tensors): All the individual losses obtained by the different loss functions
            total_loss (tf.Tensors)    : One weighted loss 
        """
        with tf.GradientTape() as tape:
            predictions = self.model(inputData, training=True)
            
            total_loss = 0
            losses = []
            for loss_fn, loss_weight in zip(self.losses_functions, self.loss_weights):
                loss = loss_fn(outputData, predictions)
                losses.append(loss)
                total_loss += loss_weight * loss
            
        if train_eval:
            gradient = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))


        return losses, total_loss
    



    def get_model(self) -> None:
        '''
        Fetching the model that should be used for training
        '''
        if self.model_name == 'cnn':
            self.model = MODELS(**self.kwargs).makeCNN()
        elif self.model_name == 'elm':
            self.model = MODELS(**self.kwargs).makeELM()
        else:
            logging.ERROR('The model_name you selected is not available.')
        
        return 
    
    def get_data(self) -> None:
        '''
        Fetching the data used for training and validation and testing.
        '''
        
        loader = DATA_LOADER(**self.kwargs)
        
        self.sensorWeights = loader.return_sensor_weights
        self.InputTrain, self.InputTest, self.OutputTrain, self.OutputTest, self.InputTrain, self.InputVal, self.OutputTrain, self.OutputVal = loader.fetch_data()
        
        return

    def piecewise_scheduler(self, epoch : int) -> float:
        '''
        Implementing a piecewise scheduler for the learning rate
        '''
        decay_rate = self.decayRate
        decay_epochs = np.arange(0, self.numIterations, self.decayStep)  # Epochs where learning rate will decrease
        learning_rate = self.initialLR
        for decay_epoch in decay_epochs:
            if epoch >= decay_epoch:
                learning_rate *= decay_rate
                
        
        return learning_rate

    def compile_model(self) -> None:
        '''
        Fetching the Loss functions from the lossFunction class
        Initializing the Learning Rate
        Initializing the Optimizer
        '''
        
        self.losses_functions  =  [self.map_losses_and_metrics(loss) for loss in self.losses]
        self.lr_schedule       = tf.keras.callbacks.LearningRateScheduler(self.piecewise_scheduler)
        self.optimizer         = tf.keras.optimizers.Adam(learning_rate = self.initialLR)

        logging.info('The Model has Compiled with metrics')
        
        return

    def log_metrics(self, losses : tf.Tensor, train_eval : bool) -> None:
        
        """
        Logs the metrics into mlflow
        """
        
        if train_eval:
            for loss, name, weight in zip(losses, self.losses, self.loss_weights):
                if weight != 0:
                    mlflow.log_metric("loss_" + name, loss.numpy(), step = self.epoch)
                else:
                    mlflow.log_metric("metric_" + name, loss.numpy(), step = self.epoch)
        else:
            for loss, name in zip(losses, self.losses):
                mlflow.log_metric('val_' + name, loss.numpy(), step = self.epoch)
                
        return
    
    def save_best_model(self, total_val_loss : float) -> None:
        """
        Saves the best tensorflow model (based on the total validation loss) into mlflow
        """
        
        # TODO
        
        if total_val_loss < self.best_val_loss: 
            mlflow.tensorflow.log_model(self.model, 'best model')
            logging.info(f'Saved a new best model at epoch: {self.epoch}')
            
        else:
            logging.info(f'No best model saved at epoch: {self.epoch}')
            
        return 
    
    def train_model(self) -> None:
        
        for self.epoch in tqdm(range(self.numIterations), desc = 'training'):
            
            train_losses, total_train_loss = self.evaluation_step(inputData=self.InputTrain, outputData=self.OutputTrain, train_eval = True)
            val_losses, total_val_loss    = self.evaluation_step(inputData=self.InputVal, outputData=self.OutputVal, train_eval = False)
            
            self.log_metrics(losses = train_losses, train_eval = True)
            self.log_metrics(losses = val_losses, train_eval = False)
            
            if self.save_best_model:
                self.save_best_model(total_val_loss = total_val_loss.numpy())
                 
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








