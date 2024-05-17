from datetime import datetime
import argparse
import os
import json
from trainer import TRAINER
import tensorflow as tf
import mlflow
import logging



if __name__ == '__main__':

    # Initialize the Logger
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force = True)
    
    
    parser = argparse.ArgumentParser(description='Train a NN')
    parser.add_argument('-p', '--processor' , type=str, nargs='+', default='cpu', help = 'What processor to use')
    args = parser.parse_args()
    
    if args.processor[0] == 'gpu':
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                logging.info(f"{len(gpus)} + Physical GPUs + {len(logical_gpus)}Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
                
    elif args.processor[0] == 'cpu':
        tf.config.set_visible_devices('CPU')

        
    else:
        logging.ERROR("Please choose either 'cpu' or 'gpu' as computation devices." )
        exit()
        
        
    # Loading in the Input Dictionary
    with open('inputDictionary.json') as config:
        inputDictionary = json.load(config)


    # Initialzing the Mlflow run
    mlflow.set_tracking_uri(inputDictionary['tracking_uri'])
    mlflow.set_experiment(inputDictionary['experiment_name'])
    
    # getting the current time to use as a run ID
    (dt, micro) = datetime.now().strftime('%Y%m%d-%H%M%S-.%f').split('.')
    inputDictionary['timestr'] = "%s%03d" % (dt, int(micro) / 1000)
    
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = inputDictionary['track_system_metrics']
    logging.info('MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING =' + inputDictionary['track_system_metrics'])
    
    with mlflow.start_run(run_name = inputDictionary['timestr']):
        
        run_id = mlflow.active_run().info.run_id
        experiment_id = mlflow.active_run().info.experiment_id
        inputDictionary['logDir'] = f"mlruns/{experiment_id}/{run_id}/artifacts"

        # Creating an Instance of our NeuralNetwork Class
        instance = TRAINER(**inputDictionary)
        
        instance.get_model()# Fetching the Model
        instance.get_data() # Fetching the Data
        instance.compile_model()# Compiling the Model
        instance.train_model()# Training the Model 
        instance.final_save()# Saving all the information
        
        logging.info('All the Files saved successfully')
    
    mlflow.end_run()
        
