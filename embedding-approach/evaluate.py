import json
import argparse
import os
import numpy as np

from train import train
from score import score
from create_db import create_db
from utils.logger import logger
from utils.dataset_utils import load_data
from utils.dataset_statistics import compute_statistics
from model.triplet import TripletLoss

def evaluate(df_train, df_db, df_eval, config, model_base_name, split_count):
        # Get statistics about dataset
        dataset_statistics = compute_statistics(df_train, df_db, df_eval, dataset_type)
        logger.info(f"Dataset has the following statistics: {dataset_statistics}")

        # train the model with params specified in config
        logger.info("Training model...")
        model_path = train(
                df=df_train,
                lr=config["train"]["learning_rate"],
                batch_size=config["train"]["batch_size"],
                input_width=config['model']['input_width'],
                input_height=config['model']['input_height'],
                embedding_size=config["model"]["embedding_size"],
                nb_epochs=config["train"]["nb_epochs"],
                sampler=config["train"]["sampler"],
                use_augmentation=config["train"]["use_augmentation"],
                augment_config=config["train"]["augment_config"],
                model_save_path=f'{config["model"]["model_save_path"]}/{dataset_path.split("/")[-1]}',
                train_val_split_overlapping=config["train"]["train_val_split_overlapping"],
                class_sampler_config = config["train"]["class_sampler_config"],
                cutoff_classes = config["model"]["cutoff_classes"],
                l2_factor = config["train"]["l2_factor"],
                img_preprocess = config["model"]["img_preprocess"],
                dataset_statistics=dataset_statistics,
                backbone = config["model"]["backbone"],
                experiment_desc = config["main"]["experiment"],
                model_base_name=model_base_name,
                split_count=split_count
        )
        
        # load the trained model to create a database of predictions
        model = TripletLoss.load_from_checkpoint(model_path)
        logger.info(f"Model trained. Stored in: {model_path}. Creating database...")
        labels, embeddings, images = create_db(
                df_train=df_train,
                model=model,
                type="database_set",
                input_width=config['model']['input_width'],
                input_height=config['model']['input_height'],
                img_preprocess=config['model']["img_preprocess"]  
        )

        # Compute the metrics for this train-db-eval split
        logger.info(f"Database created of shape {np.shape(embeddings)}. Scoring model...")
        results = score(
                df_eval=df_eval,
                model=model,
                labels=labels,
                embeddings=embeddings,
                images=images,
                input_width=config['model']['input_width'],
                input_height=config['model']['input_height'],
                img_preprocess=config['model']["img_preprocess"]
        )
        logger.info(f"Model scored. Results: {results}")
        return results