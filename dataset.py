import torch
from augments import *
from torchvision import datasets

def get_simclr_dataset(transforms, config):
    '''Makes the dataset for simclr training'''
    # Define the dict
    root_folder = './data'
    
    if config.DATASET == "cifar-10":
        dataset = datasets.CIFAR10(root_folder, train=True, transform=AugmentWithViews(transforms, config.VIEWS), download=True)
    elif config.DATASET == "stl-10":
        dataset = datasets.STL10(root_folder, split='unlabeled', transform=AugmentWithViews(transforms, config.VIEWS), download=True)
    else:
        pass
    
    # Make the dataloader and return 
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config.BATCH_SIZE,
                                         drop_last=True,
                                         num_workers=12,
                                         shuffle=True)
    
    return loader