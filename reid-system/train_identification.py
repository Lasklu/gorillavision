import argparse
import json
import os
import pytorch_lightning as pl
import pandas as pd
import torch
import numpy as np
import wandb
from gorillavision.model.triplet import TripletLoss
from gorillavision.utils.dataset_utils import load_data
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from gorillavision.utils.logger import logger
from typing import Tuple


def train(df, lr, batch_size, input_width, input_height, embedding_size, nb_epochs, sampler, use_augmentation,augment_config,
          model_save_path, train_val_split_overlapping, class_sampler_config, cutoff_classes, l2_factor, img_preprocess, dataset_statistics,
          backbone, experiment_desc="-"):
    logger.info("Initializing Model")
    img_size: Tuple[int, int] = (input_width, input_height)

    if (not os.path.exists(model_save_path)):
        logger.warning(f"Path {model_save_path} does not exist. Creating it...")
        os.mkdir(model_save_path)

    model = TripletLoss(df=df,
        embedding_size=embedding_size,
        lr=lr,
        batch_size=batch_size,
        sampler=sampler,
        use_augmentation=use_augmentation,
        augment_config=augment_config,
        train_val_split_overlapping=train_val_split_overlapping,
        class_sampler_config=class_sampler_config,
        cutoff_classes=cutoff_classes,
        l2_factor=l2_factor,
        img_size=img_size,
        img_preprocess=img_preprocess,
        backbone=backbone)

    logger.info("Initializing Wandb")
    wandb_config = {
        "learning_rate": lr,
        "embedding_size":embedding_size,
        "batch_size": batch_size,
        "max_epochs": nb_epochs,
        "sampler": sampler,
        "augmentation": use_augmentation,
        "augment_config": augment_config,
        "class_sampler_config": class_sampler_config,
        "train_val_split_overlapping": train_val_split_overlapping,
        "cutoff_classes": cutoff_classes,
        "l2_factor": l2_factor,
        "img_preprocess": img_preprocess,
        "dataset_statistics": dataset_statistics,
        "backbone": backbone,
        "experiment": experiment_desc
    }
    wandb_logger = WandbLogger(entity="mariokartrl", project="Mario-Kart-RL", config=wandb_config)
    wandb_logger.watch(model, log="all")

    logger.info("Initializing Trainer")
    checkpointCallback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="Model_"+str(wandb.run.name)+'_{epoch}-loss-{val_loss:.50f}',
        verbose=True,
        monitor='val_loss',
        mode='min')
    # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=10e-8, patience=5, verbose=False, mode="min")
    
    trainer = pl.Trainer(accelerator='gpu',
        devices=1,
        max_epochs=nb_epochs,
        logger=wandb_logger,
        callbacks=[checkpointCallback])

    logger.info("Starting Training")
    trainer.fit(model)
    logger.info("Model trained.")
    best_loss = 200000
    best_model = ""
    for model_name in list(filter(lambda file_name: file_name[:5]=="Model", os.listdir(model_save_path))):
        def get_loss(model_name):
            return float(model_name.split("=")[-1][:-5])
        def get_name(model_name):
            name = model_name.split("_")[1]
            return name
        if get_name(model_name) != str(wandb.run.name):
            continue
        if get_loss(model_name) < best_loss:
            best_loss = get_loss(model_name)
            best_model = model_name

    return os.path.join(model_save_path, best_model)

if __name__ == '__main__':
    logger.info("Loading config...")
    argparser = argparse.ArgumentParser(description='Train and validate a model on any dataset')
    argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')
    conf_name = argparser.parse_args().conf
    config_path = os.path.join("./gorillavision/configs", conf_name)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    
    df = load_data(config["train"]["dataset"]["path"])
    lr = config["train"]["learning_rate"]
    batch_size = config["train"]["batch_size"]
    input_width = config['model']['input_width']
    input_height = config['model']['input_height']
    embedding_size = config["model"]["embedding_size"]
    nb_epochs = config["train"]["nb_epochs"]
    sampler = config["train"]["sampler"]
    use_augmentation = config["train"]["use_augmentation"]
    augment_config = config["train"]["augment_config"]
    model_save_path = config["train"]["model_save_path"]
    train_val_split_overlapping = config["train"]["train_val_split_overlapping"]
    class_sampler_config = config["train"]["class_sampler_config"]
    cutoff_classes = config["model"]["cutoff_classes"]
    l2_factor = config["train"]["l2_factor"]
    img_preprocess = config["model"]["img_preprocess"]
    backbone = config["model"]["backbone"]
    model_path = train(df, lr, batch_size, input_width, input_height, embedding_size, nb_epochs, sampler, use_augmentation, augment_config,
          model_save_path, train_val_split_overlapping, class_sampler_config, cutoff_classes, l2_factor, img_preprocess, None, backbone)
    logger.info(f"Saving model under {model_path}")

    
