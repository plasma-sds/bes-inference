import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
import mlflow


class DATA_LOADER():
    
    
    def __init__(self, **kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        random.seed(self.random_seed)
        
        self.sensorWeights = np.ones(self.numPoints, dtype='float64')
        
        # Creating a Logger
        logging.basicConfig(filename=self.logDir + '/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


        
        return 
    
            
    def return_sensor_weights(self) -> np.ndarray:
        return self.sensorWeights
    
    def shape_converter(self, shape) -> np.ndarray:
        '''
        converts the shape that is stored as string to np array of floats
        shape: a string containing the shape, looks something like this: "[0.00000000e+00 4.32135816e+13 ... ]" and can contain \\n after any element of the list
        '''
        output_nparray=np.zeros(shape=(100, 4), dtype='float64')        #creating the output array    
        
        if type(shape) == str:
            rows=np.asarray(shape.split('\n')) #splitting the string that contains the whole shape to rows
            
            rows[0]=rows[0][1:]    #getting rid of "[" and "]" at the beginning and end of first and last row
            rows[-1]=rows[-1][:-1]
        else:
            rows=shape
        

        for c, row in enumerate(rows):
            elements=row.split(' ')                           #slicing the row to individual numbers
            elements=np.array(list(filter(lambda a: a != '', elements)), dtype='float64')            #dropping '' strings remaining from slicing

            output_nparray[c, :] = elements / 10**19
            
        return output_nparray.flatten()

    def get_shapes_from_df(self, df : pd.DataFrame, column_name : str) -> np.ndarray:
        '''
        converts a column of a pandas dataframe that contains shapes to a numpy array
        '''
        
        output_nparray = np.zeros(shape=(len(self.dataIndices), self.numPoints), dtype='float64')  

        for c, shape_i in enumerate(np.take(df[column_name], self.dataIndices, axis=0)):                     #adding all the shapes to the list

            output_nparray[c, :] = self.shape_converter(shape_i) 

        return output_nparray

    def fetch_data(self) -> list[tf.Tensor]:
        '''
        Fetches the Data we need for Training and Validation
        
        Returns:
        
        InputTrain   : (tf.Tensor float32 [number of Input Train Datapoints, number of Points per domain]) 
        InputTest    : (tf.Tensor float32 [number of Input Test Datapoints, number of Points per domain])  
        InputVal     : (tf.Tensor float32 [number of Input Validation Datapoints, number of Points per domain])
        OutputTrain  : (tf.Tensor float32 [number of Output Train Datapoints, number of Points per domain])  
        OutputTest   : (tf.Tensor float32 [number of Output Test Datapoints, number of Points per domain])  
        OutputVal    : (tf.Tensor float32 [number of Output Validation Datapoints, number of Points per domain])
          
        '''

        df = pd.read_hdf(self.dataDir,"df")
        
        self.dataIndices = random.sample(range(df.shape[0]), self.nData)
        
        Output = self.get_shapes_from_df(df, "Density Shape[$1/m^3$]")
        Input  = self.get_shapes_from_df(df, "Emission 2p-->2s")
        
        
        # Splitting the data into Training, Testing and Validation Sets
        InputTrain, InputTest, OutputTrain, OutputTest = train_test_split(Input, Output, test_size=self.testDataFrac, random_state=self.random_seed)
        InputTrain, InputVal, OutputTrain, OutputVal = train_test_split(InputTrain, OutputTrain, test_size=self.valDataFrac, random_state=self.random_seed)

        data     = [InputTrain, InputTest, OutputTrain, OutputTest, InputTrain, InputVal, OutputTrain, OutputVal] 
        dataname = ["InputTrain", "InputTest", "OutputTrain", "OutputTest", "InputTrain", "InputVal", "OutputTrain", "OutputVal"]
        
        # Logging the data and converting it to tensors
        for i in range(len(data)):
            mlflow.log_input(mlflow.data.from_numpy(data[i]), context = dataname[i])
            data[i]  = tf.convert_to_tensor(data[i], dtype=tf.float32)  

  
        return data