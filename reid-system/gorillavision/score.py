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
from utils.logger import logger

def score(model, image_folder, labels, embeddings, input_width, input_height, img_preprocess):
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
    for folder in tqdm(os.listdir(image_folder)):
        if os.path.isdir(os.path.join(image_folder, folder)):
            individual_name = folder
            for img_file in tqdm(os.listdir(os.path.join(image_folder, folder))):
                label, ext = os.path.splitext(img_file)
                if ext not in [".png", ".jpg", ".jpeg"]:
                    continue
                img = transform_image(imread(os.path.join(image_folder, folder, img_file)), (input_width, input_height), img_preprocess)
                with torch.no_grad():
                    predicted_embedding = model(img).numpy()
                    prediction = knn_classifier.predict(predicted_embedding)
                    neighbour_scores = knn_classifier.predict_proba(predicted_embedding)[0]
                    predicted_embeddings.append(predicted_embedding)
                    test_labels.append(individual_name)
                    predicted_labels.append(prediction)
                    predicted_scores.append(neighbour_scores)
                    all_data.append([individual_name, prediction, wandb.Image(img), *np.squeeze(predicted_embedding)])

    # Log and calculate metrics
    classes_eval = list(set(test_labels))
    print("Classes for eval: ", classes_eval)
    df = pd.DataFrame(columns=[*["target", "predicted", "img"], *dimensions], data=all_data)
    wandb.log({"val_embeddings": df})
    all_unique_labels = list(set(labels))
    all_unique_labels.sort()
    num_images_correct = sum([1 for idx in range(0, len(test_labels)) if test_labels[idx] == predicted_labels[idx]])
    print(f"correctly classified {num_images_correct}/{len(test_labels)} images")
    metrics = compute_prediction_metrics(test_labels, predicted_labels, predicted_scores, all_unique_labels)
    print("Run metrics:", metrics)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=test_labels, preds=predicted_labels)})

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
    model = TripletLoss.load_from_checkpoint(model_path)
    image_folders = config["predict"]["img_folder"]
    labels = np.load(os.path.join(config["predict"]["db_path"],'labels.npy'))
    embeddings = np.load(os.path.join(config["predict"]["db_path"],'embeddings.npy'))
    input_width = config['model']['input_width'],
    input_height = config['model']['input_height'],
    img_preprocess = config['model']["img_preprocess"]
    predict(model, image_folders, labels, embeddings, input_width, input_height, img_preprocess)
