import logging 
import os 
from src.utils.config_loader import load_config

def setup_logging(config):
    """"Sets up the logging configuration based on the provided config"""
    log_file = config['logging']['log_file']
    log_level = getattr(logging, config['logging']['log_level'].upper(), logging.INFO)

    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(f"logs/{log_file}"),
            logging.StreamHandler()
        ]
    )

    # Example of using setup_logging
if __name__ == "__main__":
    config = load_config('config/resnet-cifar100.yaml')
    setup_logging(config)
    logging.info("Logging setup is complete.")