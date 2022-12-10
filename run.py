import os
import gc
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import albumentations as alb
from datetime import datetime
from pytorch_metric_learning import losses
from torchvision import transforms as tfs
from albumentations.pytorch import ToTensorV2

from train import *
from eval import *

import warnings
warnings.simplefilter("ignore")


def RunEngine(config_train, config_eval, augment_list):
    '''
    Runs the augment selection pipeline and saves the results from the all augments
    '''
    # Define the base augments here
    base_augments = [
        alb.RandomResizedCrop(config_train.IMG_SIZE, config_train.IMG_SIZE, p=1.0),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.ColorJitter(0.6, 0.6, 0.5, (0, 0.5), p=0.7),
    ]
    # Placeholder
    df_score_global = pd.DataFrame(columns=["model_name", "augmentation_name", "eval_accuracy", "sscl_loss", "sscl_top1", "sscl_top5"])    

    # Loop and add augment and run an experiment of train and eval
    for augment_name, new_augment in augment_list.items():
        # Make a copy of base augments and add the new augments to it
        base_augments_curr = base_augments.copy()
        new_augments_curr = [new_augment, ToTensorV2(p=1.0)]
        augments_curr = alb.Compose(base_augments_curr + new_augments_curr, p=1.0)

        # Print the augment list for the current experiment
        print("############ Experiment ###############")
        print("#######################################")
        print("Augment List....")
        list_augment = [str(i).split("(")[0] for i in base_augments_curr + new_augments_curr]
        print(list_augment)
        print()

        # Change the config
        config_train.ADD_TXT = str(new_augment).split("(")[0].lower()

        # Run SSCL train and save the material
        print("Performing SSCL training.....")
        seed_everything(999)
        model_path, score_list = TrainingEngineSSCL(config_train, augments_curr)
        print()
        time.sleep(0.5)

        # Run the eval pipe to get the scores
        print("Performing SSCL evaluation.....")
        seed_everything(999)
        eval_accuracy = EvalEngineNormal(config_eval, model_path)
        print()
        time.sleep(0.5)

        # Add to the dataframe
        df_curr = pd.DataFrame({
            "model_name" : [config_train.ARCH],
            "augmentation_name" : [str(new_augment).split("(")[0].lower()],
            "eval_accuracy" : [eval_accuracy],
            "sscl_loss" : [score_list[0]],
            "sscl_top1" : [score_list[1]],
            "sscl_top5" : [score_list[2]]
        })
        df_score_global = pd.concat([df_score_global, df_curr]).reset_index(drop=True)

        print("*" * 100)
        print("*" * 100)
        print()

    # Save the dataframe
    df_score_global.to_csv(f"{config_train.SETTING}/{config_train.FOLDER_NAME}/summary.csv", index=False)
    print("Dataframe saved successfully.....")
    print("Dataframe.....")
    print(df_score_global)
    print()
    torch.cuda.empty_cache()
    return df_score_global.sort_values(by="eval_accuracy", ascending=False).reset_index().iloc[:topk], df_score_global

topk = 2
if __name__ == "__main__":
    # Define the setting
    SETTING = "HPLT" # HPLT or SSCLF
    DATASET = "cifar-10" # cifar-10 or stl-10
    ARCH_NAME = "resnet-18" # resnet-18 or resnet-50

    # Create the folder name
    now = datetime.now()
    str_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    FOLDER_NAME = f"run_{str_time}"

    # Make the augmentation list
    augment_list = [
        alb.HueSaturationValue(p=0.7),
        alb.ImageCompression(p=0.7),
        alb.ISONoise(p=0.7),
        alb.RandomFog(p=0.7),
        alb.Superpixels(p=0.7),
        alb.ChannelShuffle(p=0.7),
        alb.CLAHE(p=0.7),
        alb.Downscale(p=0.7),
        alb.Emboss(p=0.7),
        alb.Equalize(p=0.7),
        alb.FancyPCA(p=0.7),
        alb.ImageCompression(quality_lower=60, p=0.7),
        alb.JpegCompression(quality_lower=60, p=0.7),
        alb.RandomBrightnessContrast(p=0.7),
        alb.RandomGamma(p=0.7),
        alb.RandomGridShuffle(p=0.7),
        alb.RandomToneCurve(p=0.7),
        alb.RGBShift(p=0.7),
        alb.Sharpen(p=0.7),
        alb.ToGray(p=0.7),
        alb.ToSepia(p=0.7)
    ]
    augment_list = {str(i).split("(")[0].lower() : i for i in augment_list}

    # Create the folder
    newpath = os.path.join(os.getcwd(), f"{SETTING}/" + FOLDER_NAME)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Create the class config
    class GCONFIG_TRAIN:
        SETTING = SETTING
        VIEWS = 2
        IMG_SIZE = 96 if DATASET == "stl-10" else 32
        LR = 5e-4
        TEMP = 0.07
        CLASSES = 10
        DEVICE = "cuda"
        EMBED_SIZE = 512
        BATCH_SIZE = 256
        ARCH = ARCH_NAME
        TOTAL_EPOCHS = 10 if SETTING == "HPLT" else 100
        DATASET = DATASET
        ADD_TXT = ""
        FOLDER_NAME = FOLDER_NAME

    class GCONFIG_EVAL:
        SETTING = SETTING
        DEVICE = "cuda"
        TOTAL_EPOCHS = 10 if SETTING == "HPLT" else 50
        BATCH_SIZE = 128
        IMG_SIZE = 96 if DATASET == "stl-10" else 32
        ARCH = ARCH_NAME
        DATASET = DATASET
        EMBED_SIZE = 512
        PREV_FEATURES = 512 if ARCH == "resnet-18" else 2048
        VERBOSE = 1 if SETTING == "HPLT" else 10
        LR = 5e-4
        CLASSES = 10
        FOLDER_NAME = FOLDER_NAME

    best_augments, _ = RunEngine(GCONFIG_TRAIN(), GCONFIG_EVAL(), augment_list)
    print("Chosen augments....")
    print(best_augments)
    print()
    print()
    print("RUNNING FULL TRAINING WITH SELECTED AUGMENTS")
    print()


    ### FULL TRAINING ##
    ####################
    ####################

    # Define the setting
    SETTING = "SSCLF" # HPLT or SSCLF
    DATASET = "cifar-10" # cifar-10 or stl-10
    ARCH_NAME = "resnet-18" # resnet-18 or resnet-50

    # Create the folder name
    now = datetime.now()
    str_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    FOLDER_NAME = f"run_{str_time}"

    # Create the folder
    newpath = os.path.join(os.getcwd(), f"{SETTING}/" + FOLDER_NAME)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Create the class config
    class GCONFIG_TRAIN:
        SETTING = SETTING
        VIEWS = 2
        IMG_SIZE = 96 if DATASET == "stl-10" else 32
        LR = 5e-4
        TEMP = 0.07
        CLASSES = 10
        DEVICE = "cuda"
        EMBED_SIZE = 256
        BATCH_SIZE = 256
        ARCH = ARCH_NAME
        TOTAL_EPOCHS = 10 if SETTING == "HPLT" else 100
        DATASET = DATASET
        ADD_TXT = ""
        FOLDER_NAME = FOLDER_NAME

    class GCONFIG_EVAL:
        SETTING = SETTING
        DEVICE = "cuda"
        TOTAL_EPOCHS = 10 if SETTING == "HPLT" else 50
        BATCH_SIZE = 128
        IMG_SIZE = 96 if DATASET == "stl-10" else 32
        ARCH = ARCH_NAME
        DATASET = DATASET
        EMBED_SIZE = 256
        PREV_FEATURES = 512 if ARCH == "resnet-18" else 2048
        VERBOSE = 1 if SETTING == "HPLT" else 10
        LR = 5e-4
        CLASSES = 10
        FOLDER_NAME = FOLDER_NAME

    # Select the best augmentations
    best_augmentations = {i: augment_list[i] for i in best_augments["augmentation_name"]}
    _, final_scores = RunEngine(GCONFIG_TRAIN(), GCONFIG_EVAL(), augment_list)
    print("Experiment completed....")