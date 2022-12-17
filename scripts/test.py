import os

count = 0
base_dir = "/home/rohan/Downloads/data"
label_dir = "/home/rohan/Downloads/labels_good"

for file in os.listdir(base_dir):
    name, ext = os.path.splitext(file)
    if os.path.exists(os.path.join(label_dir, name + ".txt")):
        count += 1

print(count)