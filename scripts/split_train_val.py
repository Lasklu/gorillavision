import os
import random
import shutil


data_folder = ""
train_out_folder = "./train"
test_out_folder = "./val"

file_names = [name for name in os.listdir(data_folder) if os.path.isfile(name) and os.path.splitext(name)[1] == ".txt"]
amount_val = round(0.05 * len(file_names))
names = random.sample(file_names, amount_val)


for index,file in os.listdir(data_folder):
    if os.path.isfile(file):
        name, ext = os.path.splitext(name)
        if name in names:
            shutil.copyfile(os.path.join(data_folder,file), os.path.join(train_out_folder,file))
        else:
            shutil.copyfile(os.path.join(data_folder,file), os.path.join(test_out_folder,file))