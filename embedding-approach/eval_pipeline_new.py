import json
import argparse
import os
import numpy as np

from evaluate import evaluate
from utils.logger import logger

def main(dataset_paths, config):
    logger.info(f"Received the following config: {config}")
    logger.info(f"Training model in the following datasets: {dataset_paths}")
    logger.info("Initializing Wandb")
    wandb_group_id = wandb.util.generate_id()
    wandb_logger = WandbLogger(project="triplet-approach", entity="gorilla-reid", group="wandb_group_id", config=config)
    
    all_results = []
    try: 
        for dataset_path in dataset_paths:
            dataset_type = dataset_path.split("/")[-1].split("_")[0]
            logger.info(f"Creating cross validation splits with distinct={config["main"]["distinct"]}") 
            ds_splits = make_crossval_splits(dataset_path, config["main"]["distinct"])
            for split_count, ds in enumerate(ds_splits):
                df_train, df_db, df_eval = ds
                res = evaluate(df_train, df_db, df_eval, config, split_count)

    final_res = combine_results(all_results) 
    logger.info(f"All splits scored. Final results: {final_res}")

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
            logger.info(config_path)
            with open(config_path) as config_buffer:
                config = json.loads(config_buffer.read())
            dataset_paths = config['main']['datasets']
            main(dataset_paths, config)
    else:
        config_path = os.path.join("./configs", conf_name)
        with open(config_path) as config_buffer:    
            config = json.loads(config_buffer.read())
        dataset_paths = config['main']['datasets']
        main(dataset_paths, config)