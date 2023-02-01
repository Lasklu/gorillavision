import os

root_dir = '/scratch1/all_gorillas/cxl/face_images_grouped'

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        new_file_path = file_path + ".png"
        os.rename(file_path, new_file_path)