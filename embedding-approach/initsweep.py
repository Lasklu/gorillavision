import os
import yaml
from train import train
import wandb
import numpy as np
from logger import logger
from utils.dataset_utils import load_data
from utils.dataset_statistics import compute_statistics
from model.triplet import TripletLoss
from score import score
from create_db import create_db

sweep_configuration = None
with open("sweep.yaml", "r") as stream:
    sweep_configuration = yaml.safe_load(stream)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="triplet-approach")

def main():
    dataset_path = "/data/exp1/cxl_all_0_75"
    model_save_path = os.path.join("/models", dataset_path.split("/")[-1])
    run = wandb.init()
    nb_epochs = 20 if "cxl" in str(dataset_path) else 250
    augment_config = {
        "use_erase": wandb.config.use_erase,
        "use_geometric": wandb.config.use_geometric,
        "use_intensity": wandb.config.use_intensity
    }
    df = load_data(os.path.join(dataset_path, "train"))
    dataset_type = dataset_path.split("/")[-1].split("_")[0]
    dataset_statistics = compute_statistics(os.path.join(dataset_path, "train"), os.path.join(dataset_path, "database_set"), dataset_type)
    model_path = train(
                df=df,
                lr=wandb.config.lr,
                batch_size=wandb.config.batch_size,
                input_width=224,
                input_height=224,
                embedding_size=wandb.config.embedding_size,
                nb_epochs=nb_epochs,
                sampler="ensure_positive",
                use_augmentation=wandb.config.use_augmentation,
                augment_config=augment_config,
                model_save_path=model_save_path,
                train_val_split_overlapping=True,
                class_sampler_config = {},
                cutoff_classes = wandb.config.cutoff_classes,
                l2_factor = wandb.config.l2_factor,
                img_preprocess = wandb.config.img_preprocess,
                dataset_statistics=dataset_statistics,
                backbone = wandb.config.backbone
        )
    model = TripletLoss.load_from_checkpoint(model_path)
    logger.info(f"Model trained. Stored in: {model_path}. Creating database...")
    labels, embeddings, images = create_db(
            image_folder=os.path.join(dataset_path, "database_set"),
            model=model,
            type="database_set",
            input_width=wandb.config.input_width,
            input_height=wandb.config.input_height,
            img_preprocess=wandb.config.img_preprocess 
    )
    logger.info(f"Database created of shape {np.shape(embeddings)}. Scoring model...")
    results = score(
            model=model,
            image_folder=os.path.join(dataset_path, "eval"),
            labels=labels,
            embeddings=embeddings,
            images=images,
            input_width=wandb.config.input_width,
            input_height=wandb.config.input_height,
            img_preprocess=wandb.config.img_preprocess
    )
    logger.info(f"Model scored. Results: {results}")