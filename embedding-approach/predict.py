from model.triplet import TripletLoss
from sklearn.neighbors import NearestNeighbors
import torch
from cv2 import imread
import argparse
import os
import json
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')

def main():

    conf_name = argparser.parse_args().conf
    config_path = os.path.join("./configs", conf_name)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    # Load Model
    model = TripletLoss.load_from_checkpoint(config["predict"]["model_path"])
    img = imread(config["predict"]["img_path"])
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

if __name__ == "__main__":
    main()