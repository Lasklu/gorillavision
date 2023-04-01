import argparse
import os
import csv
from cv2 import imread
from gorillavision.model.triplet import TripletLoss
import torch
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from gorillavision.utils.image import transform_image
import wandb

def create_db(image_folder, model, input_width, input_height, img_preprocess, return_embedding_images=False):
    labels = []
    embeddings = []
    dimensions=[]
    
    all_data=[]
    for folder in os.listdir(image_folder):
        if folder == 'script.sh':
            continue
        for img_file in tqdm(os.listdir(os.path.join(image_folder, folder))):
            _, ext = os.path.splitext(img_file)
            if ext not in [".png", ".jpg", ".jpeg"]:
                continue
            with torch.no_grad():
                img_path = os.path.join(image_folder,folder, img_file)
                img = transform_image(imread(img_path), (input_width, input_height), img_preprocess)
                labels.append(folder)
                embedding = model(img).numpy()[0]
                embeddings.append(embedding)
                if return_embedding_images:
                    all_data.append([folder, wandb.Image(img), *embedding])
    for idx, _ in enumerate(embeddings[0]):
        dimensions.append(f"dim_{idx}")            
    embeddings_data = pd.DataFrame(data=all_data, columns=["target", "image", *dimensions])
    
    return np.array(labels), np.array(embeddings), embeddings_data

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')
    args = argparser.parse_args()
    config_path = os.path.join("./gorillavision/configs", args.conf)
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    image_folder = config["create_db"]["image_folder"]
    model = TripletLoss.load_from_checkpoint(config["create_db"]["model_path"])
    input_width = config['model']['input_width']
    input_height = config['model']['input_height']
    img_preprocess = config["model"]["img_preprocess"]
    labels, embeddings, _ = create_db(image_folder, model, input_width, input_height, img_preprocess)
    np.save(os.path.join(config["create_db"]["db_path"], "labels.npy"), np.array(labels))
    np.save(os.path.join(config["create_db"]["db_path"], "embeddings.npy"), np.array(embeddings))