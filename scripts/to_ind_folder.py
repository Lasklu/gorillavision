import os
import shutil

target = "./sorted"

for file in os.listdir("./images_cropped_to_face"):
    file_name, ext = os.path.splitext(file)
    individual = file_name.split("_")[0]
    if not os.path.exists(os.path.join(target, individual)):
        os.mkdir(os.path.join(target, individual))
    shutil.copy(os.path.join("./images_cropped_to_face", file), os.path.join(target, individual, file_name))