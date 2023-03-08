import os
import cv2
import numpy as np
import pandas as pd
import math

def load_data(dataset_path):
    individuals_df = {'images': [], 'labels': []}
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

def train_val_split_distinct(df, test_size=0.3, random_state=0, label_col_name="labels_numeric"):
    # split into train and val set without overlapping individiuals
    np.random.seed(random_state)
    individuals = df[label_col_name].unique().tolist()
    num_individuals = len(individuals)
    num_individuals_val = math.floor(test_size * num_individuals)
    np.random.shuffle(individuals)
    individuals_val = individuals[:num_individuals_val]
    individuals_train = individuals[num_individuals_val:]
    df_train = df.query(f"{label_col_name} in {individuals_train}")
    df_val = df.query(f"{label_col_name} in {individuals_val}")

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    print(f"Train test split completed. Individuals in train set: {len(individuals_train)}. Individuals in validation set: {len(individuals_val)}")
    return df_train, df_val