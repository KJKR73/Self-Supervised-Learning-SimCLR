import os
import copy
import torch
import random
import numpy as np
import pandas as pd
import albumentations as alb
from tqdm.notebook import tqdm
from torchvision import datasets
import torchvision.models as models
from sklearn.metrics import accuracy_score
from albumentations.pytorch import ToTensorV2

# Import from other files
from model import *
from utils import AverageMeter
from augments import SimpleAugment

import warnings
warnings.simplefilter("ignore")

def train_frozen_model(epoch, model, loader, loss_fxn, optimizer, config):
    '''Train the frozen model'''
    # Put the model in train mode
    model.train()
    loss_meter = AverageMeter()
    
    # Start the model training
    list_preds = []
    list_label = []
    for batch_idx, (images, labels) in enumerate(loader, 1):
        # Load the images to device
        images = images.float().to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        # Get the current batch_size
        batch_size_curr = images.shape[0]
        
        # Zero the grads  
        optimizer.zero_grad()
        
        # Collect the output
        output = model(images)
        loss = loss_fxn(output, labels)
        loss_meter.update(loss.item(), batch_size_curr)
        
        # Backward the loss
        loss.backward()
        optimizer.step()
        
        # Collect the predictions
        list_preds.append(np.argmax(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy(), axis=1))
        list_label.append(labels.detach().cpu().numpy())
    
    # Concat the preds
    list_preds = np.concatenate(list_preds)
    list_label = np.concatenate(list_label)
    
    # Print the accuracy
    accuracy_s = accuracy_score(list_label, list_preds)
    
    return round(accuracy_s, ndigits=4), loss_meter.avg

def eval_frozen_model(epoch, model, loader, loss_fxn, config):
    '''Train the frozen model'''
    # Put the model in eval mode
    model.eval()
    loss_meter = AverageMeter()
    
    # Start the model training
    list_preds = []
    list_label = []
    for batch_idx, (images, labels) in enumerate(loader, 1):
        # Load the images to device
        images = images.float().to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        # Get the current batch_size
        batch_size_curr = images.shape[0]
        
        # Collect the output
        output = model(images)
        loss = loss_fxn(output, labels)
        loss_meter.update(loss.item(), batch_size_curr)
        
        # Collect the predictions
        list_preds.append(np.argmax(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy(), axis=1))
        list_label.append(labels.detach().cpu().numpy())
    
    # Concat the preds
    list_preds = np.concatenate(list_preds)
    list_label = np.concatenate(list_label)
    
    # Print the accuracy
    accuracy_s = accuracy_score(list_label, list_preds)
    
    return round(accuracy_s, ndigits=4), loss_meter.avg


def EvalEngineNormal(config, weight_path):
    '''Train the model in normal setting'''
    # Load the model
    model = MODEL_ZOO(config)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    
    # Replace the last layer
    prev_features = config.PREV_FEATURES
    model.backbone.fc = torch.nn.Linear(in_features=prev_features,
                                        out_features=config.CLASSES)
    
    model.to(config.DEVICE)
    
    # Freeze the model
    for name, param in model.named_parameters():
        if name not in ['backbone.fc.weight', 'backbone.fc.bias']:
            param.requires_grad = False
            
    # Define the loss fxn and optimizer    
    loss_fxn = torch.nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR, weight_decay=0.0008)
    
    # Define the datasets
    root_folder = './data'
    train_dataset = datasets.CIFAR10(root_folder, train=True, transform=SimpleAugment(ToTensorV2(p=1.0)), download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                               drop_last=False, shuffle=True)
    
    val_dataset = datasets.CIFAR10(root_folder, train=False, transform=SimpleAugment(ToTensorV2(p=1.0)), download=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                                             drop_last=False, shuffle=False)
    
    # Start the training
    best_score = 0
    best_data = None
    verbose = config.VERBOSE
    for epoch in range(1, config.TOTAL_EPOCHS + 1):
        metrics = train_frozen_model(epoch=epoch, model=model,
                                     loader=train_loader, loss_fxn=loss_fxn,
                                     optimizer=optimizer, config=config)
        
        # Eval the model
        with torch.no_grad():
            eval_metrics = eval_frozen_model(epoch=epoch, model=model,
                                             loader=val_loader, loss_fxn=loss_fxn,
                                             config=config) 
            
            if epoch % verbose == 0:
                print(f"TRAIN | Epoch : {epoch} |  Accuracy : {metrics[0]} | Loss : {metrics[-1]}")
                print(f"TEST | Epoch : {epoch} | Accuracy : {eval_metrics[0]} | Loss : {eval_metrics[-1]}")
                print()
        
        # Collect the best score
        if eval_metrics[0] > best_score:
            best_score = eval_metrics[0]
            best_data = [f"TRAIN | Epoch : {epoch} |  Accuracy : {metrics[0]} | Loss : {metrics[-1]}",
                         f"TEST | Epoch : {epoch} | Accuracy : {eval_metrics[0]} | Loss : {eval_metrics[-1]}"]
            
    # Print the best accuracy
    fin_score = "\n".join(best_data)
    print()
    print(fin_score)

    return best_score


