import os
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

# Seed everything
seed_everything(999)


def RunEngine(config_train, config_eval, augment_list):
    '''
    Runs the augment selection pipeline and saves the results from the all augments
    '''
    # Define the base augments here
    base_augments = [
        alb.RandomResizedCrop(32, 32, p=1.0),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.ColorJitter(0.6, 0.6, 0.5, (0, 0.5), p=0.7),
    ]
    # Placeholder
    df_score_global = pd.DataFrame(columns=["model_name", "augmentation_name", "eval_accuracy", "sscl_loss", "sscl_top1", "sscl_top5"])    

    # Loop and add augment and run an experiment of train and eval
    for new_augment in augment_list:
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
        model_path, score_list = TrainingEngineSSCL(config_train, augments_curr)
        print()

        # Run the eval pipe to get the scores
        print("Performing SSCL evaluation.....")
        eval_accuracy = EvalEngineNormal(config_eval, model_path)
        print()

        # Add to the dataframe
        df_curr = pd.DataFrame({
            "model_name" : [config_train.ARCH],
            "augmentation_name" : [str(new_augment).split("(")[0]],
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


if __name__ == "__main__":
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
        TOTAL_EPOCHS = 10 if SETTING == "HPLT" else 100
        BATCH_SIZE = 128
        ARCH = ARCH_NAME
        DATASET = DATASET
        EMBED_SIZE = 512
        PREV_FEATURES = 512 if ARCH == "resnet-18" else 2048
        VERBOSE = 1 if SETTING == "HPLT" else 20
        LR = 5e-4
        CLASSES = 10
        FOLDER_NAME = FOLDER_NAME

    # Run the experiment
    augment_list = [
        alb.GaussianBlur(blur_limit=(1, 3), p=0.5),
        alb.Sharpen(p=0.5),
        alb.GaussNoise(p=0.5)
    ]
    RunEngine(GCONFIG_TRAIN(), GCONFIG_EVAL(), augment_list)


        



