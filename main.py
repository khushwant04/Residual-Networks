import yaml
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import os
from datetime import datetime

from src.model import ResNet50, ResNet101, ResNet152  # Update with your actual model import
from src.train import Trainer  # Assuming you save your Trainer class in src/trainer.py

def load_config(config_path='config/resenet-152-cifar100.yaml'):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_config):
    """Set up logging based on configuration."""
    log_dir = os.path.dirname(log_config['log_file'])
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        filename=log_config['log_file'],
        level=getattr(logging, log_config['log_level'].upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_config['log_level'].upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_model(model_config, device):
    """Initialize the model based on configuration."""
    model_name = model_config['name']
    num_classes = model_config['num_classes']
    input_channels = model_config['input_channels']
    
    if model_name == "ResNet50":
        model = ResNet50(img_channel=input_channels, num_classes=num_classes)
    elif model_name == "ResNet101":
        model = ResNet101(img_channel=input_channels, num_classes=num_classes)
    elif model_name == "ResNet152":
        model = ResNet152(img_channel=input_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    model = model.to(device)
    return model

def get_optimizer(optimizer_config, model_parameters):
    """Initialize the optimizer based on configuration."""
    optimizer_name = optimizer_config['name']
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")
    
    return optimizer

def get_scheduler(scheduler_config, optimizer):
    """Initialize the scheduler based on configuration."""
    scheduler_name = scheduler_config['name']
    
    if scheduler_name == "StepLR":
        step_size = scheduler_config['step_size']
        gamma = scheduler_config['gamma']
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "ReduceLROnPlateau":
        patience = scheduler_config['patience']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
    elif scheduler_name == "CosineAnnealingLR":
        T_max = scheduler_config['T_max']
        eta_min = scheduler_config.get('min_lr', 0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported scheduler name: {scheduler_name}")
    
    return scheduler

def prepare_data(data_config):
    """Prepare data loaders based on configuration."""
    dataset_name = data_config['dataset_name']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    download_path = data_config['download_path']
    shuffle = data_config.get('shuffle', True)
    validation_split = data_config.get('validation_split', 0.0)
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR100 mean and std
    ])
    
    if dataset_name == "CIFAR100":
        full_train_dataset = datasets.CIFAR100(root=download_path, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR100(root=download_path, train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    if validation_split > 0:
        val_size = int(len(full_train_dataset) * validation_split)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = None
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def main():
    # Load configuration
    config = load_config('config/resenet-152-cifar100.yaml')
    
    # Set up logging
    logger = setup_logging(config['logging'])
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    
    # Device configuration
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data(config['data'])
    logger.info(f"Number of training batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Number of validation batches: {len(val_loader)}")
    logger.info(f"Number of test batches: {len(test_loader)}")
    
    # Initialize model
    model = get_model(config['model'], device)
    logger.info(f"Initialized model: {config['model']['name']}")
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = get_optimizer(config['optimizer'], model.parameters())
    logger.info(f"Initialized optimizer: {config['optimizer']['name']} with LR: {config['optimizer']['lr']}")
    
    # Initialize scheduler if applicable
    scheduler = None
    if config['training'].get('use_scheduler', False):
        scheduler = get_scheduler(config['scheduler'], optimizer)
        logger.info(f"Initialized scheduler: {config['scheduler']['name']}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        gradient_clip_value=config['training'].get('gradient_clip_value', None),
        log_dir=config['logging']['log_dir']
    )
    
    # Start training
    num_epochs = config['training']['num_epochs']
    logger.info(f"Starting training for {num_epochs} epochs")
    trainer.train(num_epochs=num_epochs)
    logger.info("Training completed")

if __name__ == "__main__":
    main()
