import os
import cv2
import numpy as np
import pandas as pd
import math

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

def custom_train_val_split(df, test_size=0.3, random_state=0, label_col_name="labels_numeric"):
    np.random.seed(random_state)
    num_samples = len(df)
    num_samples_val = math.floor(test_size * len(df))
    label_counts = df.groupby('labels_numeric').size().reset_index(name='counts')
    classes_to_pick_from = label_counts.set_index('labels_numeric')['counts'].to_dict()

    df_val = pd.DataFrame(columns=df.columns)
    for i in range(0, num_samples_val):
        cls_keys = list(classes_to_pick_from.keys())
        if len(cls_keys) < 1:
            raise Exception("Not enough classes satisfying the criterion available")
        selected_cls = np.random.choice(cls_keys)
        sample = df.query(f"{label_col_name} == {selected_cls}").sample(random_state=random_state)
        df_val = df_val.append(sample, ignore_index=True)
        df = df.drop(sample.index)
        classes_to_pick_from[selected_cls] -= 1
        if classes_to_pick_from[selected_cls] < 3:
            classes_to_pick_from.pop(selected_cls, None)
    
    df = df.reset_index()
    df2 = df_val.reset_index()
    return df, df_val
