# bes-inference
This package features  routines for the training and development of ML based inference for BES. The main objectives of the package is multiple methods to infer beam emission from plasma density and vice-versa infer plasma density based on beam emission.



## Necessary Installation

After cloning the github repository navigate into the __/utility__ folder and clone the tfelm repository into it

```
git clone https://github.com/popcornell/tfelm.git
```

Furthermore, for now, a folder called __/data__ needs to be created in the main direcotry. This must include the **lith_dataset.hdf5** file.


## Using Mlflow

Before training any neural network you need to fix mlflow to a port, this can be done by running


```
mlflow ui --port 6006
```
in a command prompt. Before running the command navigate to the directory in which everything should be stored. Any other port can also be used. Then go and change the following in the inputDictionary.json;

1. One then needs to change the _tracking_uri_="http://127.0.0.1:6006" variable.
2. One then needs to change the _mlflow_dir_="your directory" variable.


## Environment for PINN code

This code works with a NVIDIA driver 546.33 and a NVIDIA GeForce RTX 3070. The environment can be created with python version 3.9.18 using anaconda and the **environment_pinn.yml** file.

```
conda env create -f environment_pinn.yml -n <env_name>
```

## Training a Neural Network

Navigate to the folder _/density_. With the command

```
python main_pinn.py -p <processor>
```

a neural network can be trained either on your CPU (porcessor = cpu) or GPU (processor = gpu). Using the _tracking_uri_ you can view the progress of the training in your browser of choice. 
Make changes in the _inputDictionary.json_ to change the training parameters.
