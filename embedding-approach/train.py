import argparse
import json
import os
import pytorch_lightning as pl
import pandas as pd
import cv2
import torch
import numpy as np
from model.triplet import TripletLoss
from pytorch_lightning.callbacks import ModelCheckpoint
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

def load_data(dataset_path):
    # ToDo: This loads all images into memory - is this the way to go or bad?
    individuals_df = {'images': [], 'labels': []}
    #individuals_df = pd.DataFrame(columns=['image', 'labels'])
    individuals = []

    for individual in os.listdir(dataset_path):
        folder_individual = os.path.join(dataset_path, individual)
        if os.path.isdir(folder_individual):
            individuals.append(individual)
            for img in os.listdir(folder_individual):
                img_name, ext = os.path.splitext(img)
                if ext not in [".jpg", ".jpeg", ".png"]:
                    continue
                image = cv2.imread(str(os.path.join(folder_individual, img)))
                individuals_df["images"].append(image)
                individuals_df["labels"].append(individual)
             
    name2lab = dict(enumerate(individuals))
    name2lab.update({v:k for k,v in name2lab.items()})
    individuals_df = pd.DataFrame(individuals_df)
    individuals_df["labels_numeric"] = individuals_df["labels"].map(name2lab)
    min_images_per_label = 2
    images_per_individual = np.array(np.unique(individuals_df["labels"], return_counts=True)).T
    individuals_to_remove = images_per_individual[images_per_individual[:,1] < min_images_per_label][:,0]
    individuals_df = individuals_df[~individuals_df["labels"].isin(individuals_to_remove)]
    return individuals_df

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
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=config["train"]["nb_epochs"],
        logger=logger)
        #checkpoint_callback=checkpointCallback)
    trainer.fit(model)
    
