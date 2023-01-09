from model.triplet import TripletLoss
import torch
from cv2 import imread

def main():
    # ToDo maybe in future: Preprocess images

    # Load Model
    model = TripletLoss.load_from_checkpoint(pretrained_model_path)
    img = cv2.imread(path)
    with torch.no_grad():
        embedding = model(img)

    # Load embedding_db


    # Fit knn
    #KNN.predict(db, embedding, 10)

    # Based on top-10 predictions and the according distance, apply a treshhold to check whether we are dealing 
    # with a new individual. If so, add it to db with new label.

if __name__ == "__main__":
    main()