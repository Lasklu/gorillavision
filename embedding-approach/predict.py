from model.triplet import TripletLoss
from utils.dataset_utils import load_data
from sklearn.neighbors import NearestNeighbors
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from cv2 import imread
import numpy as np
import argparse
import os
import json
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')
argparser.add_argument('-n', '--model-name', help='name of the model', default='model')

def main():
    args = argparser.parse_args()
    conf_name = args.conf
    model_name = args.model_name
    config_path = os.path.join("./configs", conf_name)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    # Load Model
    model = TripletLoss.load_from_checkpoint(f"{config['predict']['model_path']}/{model_name}")
    img = transform_image(imread(config["predict"]["img_path"]), (config['model']['input_width'], config['model']['input_height']))
    print("shape", np.shape(img))
    #img=torch.from_numpy(img).float()
    with torch.no_grad():
        predicted_embedding = model(img)

    # Load embedding_db file
    db = pd.read_csv(config["predict"]["db_path"])
    embeddings = db["label"].tolist()
    labels = db["embedding"].tolist()

    # fit kNN classifier model on db
    nn_classifier = NearestNeighbors(metric='euclidean')
    nn_classifier.fit(embeddings, labels)

    # predict k nearest neighbours
    distances, indices  = nn_classifier.kneighbors(predicted_embedding, n_neighbors=10)
    neighbour_labels = [labels[i] for i in indices]

    print(neighbour_labels)
    
    # Based on top-10 predictions and the according distance, apply a treshhold to check whether we are dealing 
    # with a new individual. If so, add it to db with new label.
    # if not return a majority vote of knn or the first as the prediction for this input

def transform_image(img, img_size):
    transformations = Compose([
            ToPILImage(),
            Resize(img_size),
            ToTensor(),
        ])
    img = transformations(img)
    return torch.unsqueeze(s, dim=0) #remove this when having multiple images

if __name__ == "__main__":
    main()