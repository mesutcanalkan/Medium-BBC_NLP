from datetime import datetime
import os
from configuration import config

def load_config_set_senv_variables():

    os.environ['logging_format'] = config.logging_config['logging_format']
    os.environ['logging_level'] = config.logging_config['logging_level']
    os.environ['logging_filemode'] = config.logging_config['logging_filemode']


    os.environ['output_folder_timestamp'] = str(datetime.now())[:19].replace(':', '.')

    os.environ['timeout'] = config.selenium_config['timeout']
    os.environ['headless'] = config.selenium_config['headless']
    os.environ['window_size'] = config.selenium_config['window_size']
    os.environ['num_retries'] = config.selenium_config['num_retries']
    

def string_to_boolean_string(input_string):
    return 'True' if str(input_string).lower().strip() in ('yes', 'true', 'tru', 'tr', 't', '1') else 'False'

def create_application_folders():
    if not os.path.exists('logs'):
        os.makedirs('logs')

    path_to_model_outputs = os.path.join(os.getcwd(), 'model_outputs', 'ModelRun' + ' ' + os.environ['output_folder_timestamp'])
    if not os.path.exists(path_to_model_outputs):
        os.makedirs(path_to_model_outputs, exist_ok=True)