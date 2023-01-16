import os
import random
import shutil


data_folder = "./data/cropped"
train_out_folder = "./data/train"
test_out_folder = "./data/test"
eval_folder = "./data/eval"

def sufficient_images(individual_name, amount):
    # check if at least amount images are available for the individual
    folder_path = os.path.join(data_folder, individual_name)
    return len(os.listdir(folder_path)) >= amount

def copy_to(individuals, dest_folder):
    # copies all these individuals to the train folder
    for individual in individuals:
        shutil.copytree(os.path.join(data_folder,individual), os.path.join(dest_folder,individual))

individuals = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder,name)) and sufficient_images(name, 3)]
image_count = sum([1 for individual in individuals for s in os.listdir(os.path.join(data_folder, individual))])
print("Total images after removing individuals with to little samples: ", image_count)
amount_test = round(0.1 * len(individuals))
random.shuffle(individuals)
copy_to(individuals[:amount_test], test_out_folder)
copy_to(individuals[amount_test:], train_out_folder)
amount_eval = round(0.05 * sum([1 for individual in os.listdir(test_out_folder) for s in os.path.join(test_out_folder, individual)]))
print("Eval:", amount_eval)

individuals_test = individuals[:amount_test]
if not os.path.exists(eval_folder):
    os.mkdir(eval_folder)
for i in range(0, amount_eval):
    while True:
        individual = random.sample(individuals_test, 1)[0]
        individual_path = os.path.join(test_out_folder, individual)
        # only pick sample if afterwards more than 3 images still left for the individual
        if len(os.listdir(individual_path)) < 4:
            continue
        individual_samples = [name for name in os.listdir(individual_path) if os.path.splitext(name)[1] in [".jpg", ".jpeg", ".png"]]
        sample = random.sample(individual_samples, 1)[0]
        if not os.path.exists(os.path.join(eval_folder, individual)):
            os.mkdir(os.path.join(eval_folder, individual))
        shutil.move(os.path.join(individual_path, sample), os.path.join(eval_folder, individual, sample))
        break
