from torchvision import transforms
import logging 

logger = logging.getLogger(__name__)

def get_transforms(train=True):
    if train:
        logger.info("Getting training set transformations.")
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
    else:
        logger.info("Getting testing transformations.")
        return transforms.Compose([
            transforms.ToTensor(),               
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)) 
        ])