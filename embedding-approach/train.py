import argparse
import json
import os
import pytorch_lightning as pl
import pandas as pd
import cv2
import torch
from cv2 import imread
import numpy as np
from model.triplet import TripletLoss
from utils.dataset_utils import load_data
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Tuple
###############################
#  Setup Argparse
###############################

argparser = argparse.ArgumentParser(description='Train and validate a model on any dataset')
argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')


###############################
#  Load Data
###############################


###############################
#  Run application
###############################

if __name__ == '__main__':
    conf_name = argparser.parse_args().conf
    config_path = os.path.join("./configs", conf_name)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    img_size: Tuple[int, int] = (config['model']['input_width'], config['model']['input_height'])
    df = load_data(config["data"]["path"])#, img_size)

    model = TripletLoss(df=df,
        embedding_size=config["model"]["embedding_size"],
        lr=config["train"]["learning_rate"],
        img_size=img_size)
    logger = TensorBoardLogger("./tensorboard", name="reID-model")
    checkpointCallback = ModelCheckpoint(
        dirpath=config["model"]["model_save_path"],
        filename='{epoch}-{val_loss:.2f}',
        verbose=True,
        monitor='val_loss',
        mode='min')
    #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=10e-8, patience=3, verbose=False, mode="min")

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=config["train"]["nb_epochs"],
        logger=logger,
        callbacks=[checkpointCallback])
    print("training")
    trainer.fit(model)
    
