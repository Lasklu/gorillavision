import argparse
from cv2 import imread
import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score,top_k_accuracy_score
from utils.metrics import compute_prediction_metrics
from model.triplet import TripletLoss
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from cv2 import imread
import numpy as np
from cv2 import imread
import argparse
import os
import wandb
import json
from utils.image import transform_image

def score(df_eval, model, labels, embeddings, images, input_width, input_height, img_preprocess):
    # setup knn classifier based in labels and embeddings from "db", k=5 default
    knn_classifier = neighbors.KNeighborsClassifier()
    knn_classifier.fit(embeddings, labels)

    # predict embedding for every image in image_folder
    predicted_embeddings = []
    predicted_labels = []
    predicted_scores = []
    test_labels = []
    dimensions = []
    for idx, _ in enumerate(embeddings[0]):
        dimensions.append(f"dim_{idx}")
    all_data = []
    for index, row in df_eval.iterrows():
        img = transform_image(row["images"], (input_width, input_height), img_preprocess)
        with torch.no_grad():
                predicted_embedding = model(img).numpy()
                prediction = knn_classifier.predict(predicted_embedding)
                neighbour_scores = knn_classifier.predict_proba(predicted_embedding)[0]
                predicted_embeddings.append(predicted_embedding)
                test_labels.append(row["labels"])
                predicted_labels.append(prediction)
                predicted_scores.append(neighbour_scores)
                all_data.append([row["labels"], prediction, wandb.Image(img), *np.squeeze(predicted_embedding)])

    # Log and calculate metrics
    df = pd.DataFrame(columns=[*["target", "predicted", "img"], *dimensions], data=all_data)
    wandb.log({"eval_embeddings": df})
    metrics = compute_prediction_metrics(test_labels, predicted_labels, predicted_scores, list(set(labels)))
    logger.info(metrics)
    for metric, value in metrics.items():
        wandb.summary[metric] = value
    wandb.finish()

    return metrics


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')
    args = argparser.parse_args()
    conf_name = args.conf
    config_path = os.path.join("./configs", conf_name)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    model_path = config['predict']['model_path']
    image_folders = config["predict"]["img_folder"]
    labels = np.load(os.path.join(config["predict"]["db_path"],'labels.npy'))
    embeddings = np.load(os.path.join(config["predict"]["db_path"],'embeddings.npy'))
    predict(model_path, image_folders, labels, embeddings)
