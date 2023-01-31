import os
from PIL import Image

def read_labels(file_path):
    labels = []
    with open(file_path) as label_file:
        for line in label_file:
            labels.append(line)
    return labels

def yolobbox2bbox(x,y,w,h, img_w, img_h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1*img_w, y1*img_h, x2*img_w, y2*img_h

images_folder = "/home/rsawahn/data/bristol_all"
lables_folder = images_folder
output_folder = os.path.join(images_folder, "cropped")

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for file in os.listdir(images_folder):
    file_name, file_extension = os.path.splitext(file)
    label_path = os.path.join(lables_folder, file_name + ".txt")
    if file_extension != ".png" or not os.path.exists(label_path):
        continue
    labels = read_labels(label_path)
    print(len(labels))
    for label in labels:
        new_folder = os.path.join(output_folder, str(label.split(" ")[0]))
        if not os.path.exists(os.path.join(new_folder)):
            os.mkdir(new_folder)
        img = Image.open(os.path.join(images_folder, file))
        bbox = label.split()[1:]
        bbox = yolobbox2bbox(*bbox, img.width, img.height)
        cropped_image = img.crop(tuple(bbox))
        new_path = os.path.join(new_folder, file_name + ".png")
        cropped_image.save(new_path)