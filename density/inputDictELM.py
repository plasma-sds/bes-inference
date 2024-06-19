from dataclasses import dataclass, field
import os
import tensorflow as tf
from sklearn.linear_model import Ridge
# Append the path of the utility folder to sys.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utility.lossFunctions import LOSSES

@dataclass
class inputDictionary:
    
    logDir              : str = ""
    timestr             : str = ""
    mlflow_dir          : str = "C:/Users/karac/AppData/Roaming/Python/Python38/Scripts"
    dataDir             : str = "C:/Users/karac/elte/bme_kutatas/neuraldiagnostics/oldnetwork_flucdata/lith_dataset.hdf5"
    tracking_uri        : str = "http://127.0.0.1:8080"
    experiment_name     : str = "BES"
    track_system_metrics: str = "True"
    model_name          : str = 'elm'
    numLayers           : int = 2
    layerType           = ["BatchIntrinsicPlasticity","InputToNode"]
    layerName           = ["BIP","i2n"]
    numNodes            = [500,1000]
    activationFnc       = ["tanh","logistic"]
    inScale             = ["N/A",0.75]
    inShift             = ["N/A",0.0]
    biScale             = ["N/A",0.0]
    biShift             = ["N/A",0.0]
    distribution        = ["exponential","N/A"]
    algorithm           = ["dresden","N/A"]
    regressor           = Ridge(alpha=1e-06)
    losses              = [LOSSES().eval_mae, LOSSES().eval_mape, LOSSES().eval_mse, LOSSES().eval_rce]
    losses_str          = ["MeanAbsoluteError", "MeanAveragePercentageError", "MeanSquaredError", "RelativeCumulativeError"]
    loss_weights        = [0.0, 0.0, 0.0, 1.0]
    save_best_model     : bool = False
    nData               : int = 60000
    testDataFrac        : float = 0.2
    valDataFrac         : float = 0.2
    numPoints           : int = 400
    random_seed         : int = 42


