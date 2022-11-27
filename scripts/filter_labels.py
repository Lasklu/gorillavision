import os
import shutil

root_dir = "/Users/lukaslaskowski/Downloads/Individual_ID"
label_dir = "/Users/lukaslaskowski/Downloads/data/labels"
new_img_folder = "/Users/lukaslaskowski/Downloads/data/images"

m = dict()

for f in os.listdir(label_dir):
    filename, extension = os.path.splitext(f)
    print(filename)
    print(filename.split("-"))
    prefix, original_name = filename.split("-")
    m[original_name] = prefix

for dir in os.listdir(root_dir):
    for f in os.listdir(os.path.join(root_dir, dir)):
        filename, extension = os.path.splitext(f)
        if filename in m:
            src = os.path.join(root_dir, dir, f)
            new_file_name = m[filename] + "-" +  filename
            dst = os.path.join(root_dir, new_img_folder, new_file_name + '.png')
            shutil.copy(src, dst)