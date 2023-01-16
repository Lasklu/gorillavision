import os
import csv
from cv2 import imread
from model.triplet import TripletLoss
import torch
import json
import pandas as pd
import numpy as np
from utils.image import transform_image
def main():
    image_folder = "/data/data/test"
    db_file = "/gorilla-reidentification/embedding-approach/database"
    pretrained_model_path = "/data/models/epoch=273-val_loss=0.00.ckpt"
    config_path = "/gorilla-reidentification/embedding-approach/configs/config.json"
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    model = TripletLoss.load_from_checkpoint(pretrained_model_path)
    labels = []
    embeddings = []
    for folder in os.listdir(image_folder):
        if folder == 'script.sh':
            continue
        for img_file in os.listdir(os.path.join(image_folder, folder)):
            label, ext = os.path.splitext(img_file)
            if ext not in [".png", ".jpg", ".jpeg"]:
                continue
            with torch.no_grad():
                img_path = os.path.join(image_folder,folder, img_file)
                img = transform_image(imread(img_path), (config['model']['input_width'], config['model']['input_height']))
                labels.append(label[:4])
                embeddings.append(model(img).numpy()[0])
    np.save(os.path.join(config["predict"]["db_path"], "database/labels.npy"), np.array(labels))
    np.save(os.path.join(config["predict"]["db_path"], "database/embeddings.npy"), np.array(embeddings))
    

if __name__ == '__main__':
    main()