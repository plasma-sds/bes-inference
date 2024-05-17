from dataclasses import dataclass, field
import os
import tensorflow as tf
# Append the path of the utility folder to sys.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utility.lossFunctions import LOSSES

@dataclass
class inputDictionary:
    
    logDir            : str = ""
    timestr           : str = ""
    mlflow_dir        : str = "/mnt/c/Users/leona/Desktop/code/ElectronDensityReconstruction/NN/"
    dataDir           : str ="/mnt/c/Users/leona/Desktop/code/bes-inference/data/lith_dataset.hdf5"
    tracking_uri      : str = "http://127.0.0.1:6006"
    experiment_name   : str = "BES"
    track_sys_metric  : str = "True"
    model_name        : str = "cnn"
    initialLR         : float = 0.07
    lr_schedule       = tf.keras.optimizers.schedules.PiecewiseConstantDecay([5, 7], [0.001, 0.002, 0.003], name='PiecewiseConstant')
    optimizer         = tf.keras.optimizers.Adam(learning_rate = initialLR)
    activationFnc     : str = "tanh"
    numLayers         : int = 8
    numNodes          : int = 50
    dropoutFreq       : float = 0.1
    losses            = [LOSSES().eval_mae, LOSSES().eval_mape, LOSSES().eval_mse]
    losses_str        = ["MeanAbsoluteError", "MeanAveragePercentageError", "MeanSquaredError"]
    loss_weights      = [1.0, 1.0, 0.0]
    save_best_model   : bool = False
    nData             : int = 30000
    testDataFrac      : float = 0.1
    valDataFrac       : float = 0.1
    numIterations     : int = 20
    batchSize         : int = 800
    decayRate         : float = 0.3
    decayStep         : int = 150
    numPoints         : int = 400
    random_seed       : int = 42
  
  

