# config_loader.py
import yaml

def load_config(config_path='config/resnet-cifar100.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
