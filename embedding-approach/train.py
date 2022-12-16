import argparse
import json
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import pandas as pd
import cv2
import torch
from model.triplet import TripletLoss

###############################
#  Setup Argparse
###############################

argparser = argparse.ArgumentParser(description='Train and validate a model on any dataset')
argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')


###############################
#  Load Data
###############################

def load_data(dataset_path):
    individuals_df = pd.DataFrame()

    for individual in os.listdir(dataset_path):
        folder_individual = os.path.join(dataset_path, individual)
        if os.path.isdir(folder_individual):
            for img in os.listdir(folder_individual):
                img_name, ext = os.path.splitext(img)
                if ext not in [".jpg", ".jpeg", ".png"]:
                    continue
                image = cv2.imread(str(os.path.join(folder_individual, img)))
                individuals_df.append({
                    "image": image,
                    "label": individual
                })
                
    return individuals_df


if __name__ == '__main__':
    conf_name = argparser.parse_args().conf
    config_path = os.path.join("./configs", conf_name)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    df = load_data(config["data_path"])

    model = TripletLoss(df)
    logger = TensorBoardLogger("covid-mask-detector/tensorboard", name="mask-detector")
    checkpointCallback = ModelCheckpoint(
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=10,
                         logger=logger,
                         checkpoint_callback=checkpointCallback)
    trainer.fit(model)