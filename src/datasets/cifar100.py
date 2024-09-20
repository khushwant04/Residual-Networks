import logging
from torchvision import datasets
from torch.utils.data import DataLoader
from src.utils.config_loader import load_config
from src.transforms import get_transforms

logger = logging.getLogger(__name__)

def get_dataloaders(config_path='config/resnet-cifar100.yaml'):
    config = load_config(config_path)

    # Extract configuration
    dataset_name = config['data']['dataset_name']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    download_path = config['data']['download_path']

    if dataset_name != 'CIFAR100':
        raise ValueError(f"Expected dataset name 'CIFAR100', got '{dataset_name}' instead.")

    logger.info('Loading CIFAR-100 dataset with the following parameters:')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Number of workers: {num_workers}')
    logger.info(f'Download path: {download_path}')

    # Load the train dataset
    train_dataset = datasets.CIFAR100(
        root=download_path,
        train=True,
        download=True,
        transform=get_transforms(train=True)
    )

    # Load the test dataset
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

    logger.info(f'Training dataset size: {len(train_dataset)} samples.')
    logger.info(f'Validation dataset size: {len(val_dataset)} samples.')

    return train_loader, val_loader


def main():
    from src.utils.resnet_logging import setup_logging
    config = load_config('config/resenet-101-cifar100.yaml')
    setup_logging(config)

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders()

    # Print one tensor from the train_loader
    for images, labels in train_loader:
        print(f"Image tensor shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Print only one batch and exit loop

if __name__ == "__main__":
    main()
