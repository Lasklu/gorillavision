import os
import cv2
import numpy as np
import pandas as pd
def load_data(dataset_path):
    # ToDo: This loads all images into memory - is this the way to go or bad?
    individuals_df = {'images': [], 'labels': []}
    #individuals_df = pd.DataFrame(columns=['image', 'labels'])
    individuals = []

    for individual in os.listdir(dataset_path):
        folder_individual = os.path.join(dataset_path, individual)
        if os.path.isdir(folder_individual):
            individuals.append(individual)
            for img in os.listdir(folder_individual):
                img_name, ext = os.path.splitext(img)
                if ext not in [".jpg", ".jpeg", ".png"]:
                    continue
                image = cv2.imread(str(os.path.join(folder_individual, img)))
                individuals_df["images"].append(image)
                individuals_df["labels"].append(individual)
             
    name2lab = dict(enumerate(individuals))
    name2lab.update({v:k for k,v in name2lab.items()})
    individuals_df = pd.DataFrame(individuals_df)
    individuals_df["labels_numeric"] = individuals_df["labels"].map(name2lab)
    min_images_per_label = 2
    images_per_individual = np.array(np.unique(individuals_df["labels"], return_counts=True)).T
    individuals_to_remove = images_per_individual[images_per_individual[:,1] < min_images_per_label][:,0]
    individuals_df = individuals_df[~individuals_df["labels"].isin(individuals_to_remove)]
    return individuals_df