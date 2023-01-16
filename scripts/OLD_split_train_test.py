import os
import random
import shutil


data_folder = ""
train_out_folder = "./train"
test_out_folder = "./val"

def sufficient_images(individual_name, amount):
    # check if at least amount images are available for the individual
    folder_path = os.path.join(data_folder, individual_name)
    return len(os.path.listdir(folder_path)) >= amount

def copy_to_train(individuals):
    # copies all these individuals to the train folder
    for individual in individuals:
        shutil.copytree(os.path.join(data_folder,individual), os.path.join(train_out_folder,individual))

individuals = [name for name in os.listdir(data_folder) if os.path.isdir(name) and sufficient_images(name, 3)]
image_count = [1 for s in os.listdir(os.path.join(data_folder, individual)) for individual in individuals].sum()
print("Total images after removing individuals with to little samples: ", image_count)
amount_test = round(0.05 * len(individuals))
copy_to_train(individuals)

for i in range(0, amount_test):
    while True:
        individual = random.sample(individual, 1)[0]
        individual_path = os.path.join(train_out_folder, individual_name)
        # only pick sample if afterwards more than 3 images still left for the individual
        if len(os.listdir(individual_path)) < 4:
            continue
        individual_samples = [name for name in os.listdir(individual_path) if os.path.splitext(name)[1] is in [".jpg", ".jpeg", ".png"]]
        sample = random.sample(individual_samples, 1)[0]
        if not os.path.exists(os.path.join(test_out_folder, individual)):
            os.mkdir(os.path.join(test_out_folder, individual))
        shutil.move(os.path.join(individual_path, sample), os.path.join(test_out_folder, individual, sample))

image_count_train = [1 for s in os.listdir(os.path.join(train_out_folder, individual)) for individual in os.path.listdir(train_out_folder)].sum()
print("Total images in train: ", image_count_train)
image_count_test = [1 for s in os.listdir(os.path.join(test_out_folder, individual)) for individual in os.path.listdir(test_out_folder)].sum()
print("Total images in test: ", image_count_test)
if image_count_test + image_count_train != image_count:
    raise Exception("Train test split sum does not match total images")