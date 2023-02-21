import json
import argparse
import os
import numpy as np

from train import train
from score import score
from create_db import create_db
from utils.logger import logger
from utils.dataset_utils import load_data
from model.triplet import TripletLoss
from utils.dataset_statistics import compute_statistics

def main(dataset_paths, config):
    logger.info(f"Received the following config: {config}")
    logger.info(f"Training model in the following datasets: {dataset_paths}")
    try: 
        for dataset_path in dataset_paths:
            dataset_type = dataset_path.split("/")[-1].split("_")[0]
            logger.info(f"Loading data for dataset of type {dataset_type} stored in: {dataset_path}...")
            dataset_statistics = compute_statistics(os.path.join(dataset_path, "train"), os.path.join(dataset_path, "database_set"), os.path.join(dataset_path, "eval"), dataset_type)
            logger.info(f"Dataset has the following statistics: {dataset_statistics}")
            df = load_data(os.path.join(dataset_path, "train"))
            logger.info("Training model...") 

            model_path = train(
                    df=df,
                    lr=config["train"]["learning_rate"],
                    batch_size=config["train"]["batch_size"],
                    input_width=config['model']['input_width'],
                    input_height=config['model']['input_height'],
                    embedding_size=config["model"]["embedding_size"],
                    nb_epochs= int(config["train"]["nb_epochs"] / 4) if "bristol" in dataset_path else config["train"]["nb_epochs"],
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
                    experiment_desc = config["main"]["experiment"]
            )
            model = TripletLoss.load_from_checkpoint(model_path)
            logger.info(f"Model trained. Stored in: {model_path}. Creating database...")
            labels, embeddings, images = create_db(
                    image_folder=os.path.join(dataset_path, "database_set"),
                    model=model,
                    type="database_set",
                    input_width=config['model']['input_width'],
                    input_height=config['model']['input_height'],
                    img_preprocess=config['model']["img_preprocess"]  
            )
            logger.info(f"Database created of shape {np.shape(embeddings)}. Scoring model...")
            results = score(
                    model=model,
                    image_folder=os.path.join(dataset_path, "eval"),
                    labels=labels,
                    embeddings=embeddings,
                    images=images,
                    input_width=config['model']['input_width'],
                    input_height=config['model']['input_height'],
                    img_preprocess=config['model']["img_preprocess"]
            )
            logger.info(f"Model scored. Results: {results}")
    except Exception as Argument:
        logger.exception("Sorry, but an error occured. Seems like the gorillas do not want to be identified: Fix your code;)")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')
    argparser.add_argument('-d','--dir', help='use multiple configs from directory, path to directy', default=None)
    args = argparser.parse_args()
    conf_name = args.conf
    config_folder = args.dir
    if args.dir:
        for conf_name in os.listdir(config_folder):
            config_path = os.path.join(config_folder, conf_name)
            print(config_path)
            with open(config_path) as config_buffer:    
                config = json.loads(config_buffer.read())
            base_path = config['main']['path']
            dataset_paths = [os.path.join(base_path, dataset) for dataset in os.listdir(base_path)]
            main(dataset_paths, config)
    else:
        config_path = os.path.join("./configs", conf_name)
        with open(config_path) as config_buffer:    
            config = json.loads(config_buffer.read())
        base_path = config['main']['path']
        dataset_paths = [os.path.join(base_path, dataset) for dataset in os.listdir(base_path)]
        main(dataset_paths, config)