import os
import shutil

target = "./data/cxl/body_images_grouped"

for file in os.listdir("./data/cxl/body_images/"):
    file_name, ext = os.path.splitext(file)
    individual = file_name.split("_")[0]
    if not os.path.exists(os.path.join(target, individual)):
        os.mkdir(os.path.join(target, individual))
    shutil.copy(os.path.join("./data/cxl/body_images", file), os.path.join(target, individual, file_name))