import argparse
from cv2 import imread
import json
import os
import pandas as pd
import numpy as np
from sklearn import neighbors
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from cv2 import imread
import numpy as np
from cv2 import imread
import argparse
import os
import json
from utils.image import transform_image

argparser = argparse.ArgumentParser()
argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')

def main():

    # Load config
    args = argparser.parse_args()
    conf_name = args.conf
    model_name = args.model_name
    config_path = os.path.join("./configs", conf_name)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    # Load Model
    model = TripletLoss.load_from_checkpoint(f"{config['predict']['model_path']}/{model_name}")
    #img = transform_image(imread(config["predict"]["img_path"]), (config['model']['input_width'], config['model']['input_height']))
    #img=torch.from_numpy(img).float()
    #with torch.no_grad():
    #    predicted_embedding = model(img)

    # Load Model and DB and fit kNN-classifier on DB
    model = TripletLoss.load_from_checkpoint(f"{config['predict']['model_path']}")
    labels = np.load(os.path.join(config["predict"]["db_path"],'labels.npy'))
    embeddings = np.load(os.path.join(config["predict"]["db_path"],'embeddings.npy'))
    
    # fit kNN classifier model on db
    nn_classifier = neighbors.KNeighborsClassifier()
    print("training")
    nn_classifier.fit(embeddings, labels)
    test_labels = []
    test_embeddings = []
    image_folder = "/data/data/gorillas_dante/face_cropped/train_test/test"
    for folder in os.listdir(image_folder):
        if folder == 'script.sh':
            continue
        print(folder)
        for img_file in os.listdir(os.path.join(image_folder, folder)):
            label, ext = os.path.splitext(img_file)
            if ext not in [".png", ".jpg", ".jpeg"]:
                continue
            with torch.no_grad():
                img_path = os.path.join(image_folder,folder, img_file)
                img = transform_image(imread(img_path), (config['model']['input_width'], config['model']['input_height']))
                test_labels.append(folder)
                print("transformed, applying model")
                test_embeddings.append(model(img).numpy()[0])
    test_labels = np.array(test_labels)
    test_embeddings = np.array(test_embeddings)
    print("scoring")
    score = nn_classifier.score(test_embeddings, test_labels)
    print("scored", score)
    # predict k nearest neighbours
    
    #distances, indices  = nn_classifier.kneighbors(predicted_embedding, n_neighbors=10)
    #neighbour_labels = [labels[i] for i in indices]

    total_predictions = []

    # predict embedding for every image in image_folder specified in config
    for folder is os.listdir(config["predict"]["img_folder"]):
        if os.isdir(folder):
            individual_name = folder
            for img_file in os.listdir(os.path.join(config["predict"]["img_folder"], folder)):
                label, ext = os.path.splitext(img_file)
                if ext not in [".png", ".jpg", ".jpeg"]:
                    continue
                img = transform_image(imread(config["predict"]["img_path"]), (config['model']['input_width'], config['model']['input_height']))
                with torch.no_grad():
                    predicted_embedding = model(img)
                    predicted_label = nn_classifier.predict(predicted_embedding.numpy(), )
                    neighbour_prediction = nn_classifier.kneighbors(predicted_embedding.numpy())
                    total_predictions.append({"prediction": neighbour_prediction, "predicted__label": predicted__label, "real_label": label})
    
    metrics = compute_prediction_metrics(total_predictions)
    print(metrics)


if __name__ == "__main__":
    main()
