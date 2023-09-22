import configparser
import os.path

config = configparser.ConfigParser()
config.read('config/secrets.ini')
dataset_base_dir = config['dataset']['base_dir']
dataset_cpol_dir = os.path.join(config['dataset']['base_dir'], config['dataset']['kdprain'])
