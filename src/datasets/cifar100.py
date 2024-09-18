from sympy import true
import logging
from src.transforms import get_transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from src.utils.config_loader import load_config


logger = logging.getLogger(__name__)

def get_dataloaders(config_path='config/resnet-cifar100.yaml'):
    config = load_config(config_path)

    # Extract configuration
    dataset_name = config['data']['dataset_name']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    download_path = config['data']['download_path']

    if dataset_name != 'CIFAR100':
        raise ValueError("Please the dataset configurations.")
    
    logger.info('Loading CIFAR-100 dataset with the following parameters: ')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f"Number of workers: {num_workers}")
    logger.info(f"Download Path: {download_path}")

    # load the Train dataset
    train_dataset = datasets.CIFAR100(
        root=download_path,
        train=True,
        download=True,
        transform=get_transforms(train=True)
    )

    # load the Test dataset
    val_dataset = datasets.CIFAR100(
        root=download_path,
        train=False,
        download=True,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


    logger.info(f"Training dataset size: {len(train_dataset)} samples.")
    logger.info(f"Validation dataset size: {len(val_dataset)} samples.")

    return train_loader, val_loader

if __name__ == "__main__":
    get_dataloaders()

    
