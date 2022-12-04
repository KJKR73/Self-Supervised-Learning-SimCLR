import os
import torch
import random
import numpy as np
from tqdm import tqdm
import albumentations as alb
from pytorch_metric_learning import losses
from torchvision import transforms as tfs
from albumentations.pytorch import ToTensorV2

from utils import *
from losses import *
from model import *
from dataset import  *

import warnings
warnings.simplefilter("ignore")

# Define the seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ss_one_epoch(epoch, model, scaler, loader, loss_fxn, optimizer, config):
    '''self supervise one epoch'''
    # Put the model in training mode
    model.train()
    loss_meter = AverageMeter()
    average_top1 = AverageMeter()
    average_top5 = AverageMeter()
    
    # Start the model training 
    if config.SETTING == "HPLT":
        bar = tqdm(loader, total=10)
    else:
        bar = tqdm(loader, total=len(loader))
    
    # Loop and train the model
    for batch_idx, (img_out, _) in enumerate(bar, 1):
        # Collect the images
        images = torch.cat(img_out, dim=0).float().to(config.DEVICE)
        
        # Get the current batch_size
        batch_size_curr = images.shape[0]
        
        # Enable fp-16 training
        with torch.cuda.amp.autocast(enabled=True):
            embeds = model(images)
            logits, labels = info_nce_loss(embeds, config)
            loss = loss_fxn(logits, labels)
        
        # Update the loss
        loss_meter.update(loss.item(), batch_size_curr)
            
        # Backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Get the average
        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        average_top1.update(top1.item(), batch_size_curr)
        average_top5.update(top5.item(), batch_size_curr)
        
        # Print the loss on the go
        bar.set_description(f"Epoch : {epoch} | Loss : {round(loss_meter.avg, ndigits=4)} | Acc-top1 : {round(average_top1.avg,ndigits=4)} | Acc-top5 : {round(average_top5.avg, ndigits=4)}")

        # Stop the data
        if config.SETTING == "HPLT":
            if batch_idx == 11:
                break
        
    return loss_meter.avg, average_top1.avg, average_top5.avg


def TrainingEngineSSCL(config, transforms):
    '''Full training pipeline for the model'''
    # Define the model
    model = MODEL_ZOO(config)
    model.to(config.DEVICE)
        
    # Get the dataset and loader
    loader = get_simclr_dataset(transforms, config)
    warmup_epochs = 2 if config.SETTING == "HPLT" else 10
    
    # Optimizer and loss functions
    scaler = torch.cuda.amp.GradScaler()
    loss_fxn = torch.nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader), eta_min=0,
                                                           last_epoch=-1)
    
    # Start the training
    list_score = []
    for epoch in range(1, config.TOTAL_EPOCHS + 1):
        metrics = ss_one_epoch(epoch=epoch, model=model, scaler=scaler,
                               loader=loader, loss_fxn=loss_fxn,
                               optimizer=optimizer, config=config)
        
        # warmup for the first 10 epochs
        if epoch >= warmup_epochs:
            scheduler.step()

        # Save the model
        torch.save(model.state_dict(), f"{config.SETTING}/{config.FOLDER_NAME}/weight_{config.ARCH}_{config.ADD_TXT}.pth")

        # Append the data to the list
        list_score.append(f"Top 1 : {metrics[1]} | Top 5 : {metrics[-1]} | Loss : {metrics[0]}")
    
    # Get the score of the last epoch
    last_epoch_score = [metrics[0], metrics[1], metrics[-1]]
    
    # Add to the report
    with open(f"{config.SETTING}/{config.FOLDER_NAME}/log_{config.ARCH}_{config.ADD_TXT}.txt", "w") as f:
        f.write("\n".join(list_score))

    return f"{config.SETTING}/{config.FOLDER_NAME}/weight_{config.ARCH}_{config.ADD_TXT}.pth", last_epoch_score