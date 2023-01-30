import os
from PIL import Image

def read_label(file_path, id):
    with open(file_path) as label_file:
        for line in label_file:
            if line.split(" ")[0] == id:
                corners = line.split()[1:]
                return [float(val) for val in corners]
    return None

def yolobbox2bbox(x,y,w,h, img_w, img_h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1*img_w, y1*img_h, x2*img_w, y2*img_h

label_id = "0" # descripes which label should be read. Currently there must be only one label of this type
images_folder = "./data/cxl/images/all_images"
lables_folder = "./data/cxl/labels"
output_folder = os.path.join(images_folder, "cropped")

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for file in os.listdir(images_folder):
    file_name, file_extension = os.path.splitext(file)
    label_path = os.path.join(lables_folder, file_name + ".txt")
    if file_extension != ".png" or not os.path.exists(label_path):
        continue
    bbox = read_label(label_path, label_id)
    if not bbox:
        continue
    img = Image.open(os.path.join(images_folder, file))
    bbox = yolobbox2bbox(*bbox, img.width, img.height)
    cropped_image = img.crop(tuple(bbox))
    new_path = os.path.join(output_folder, file_name + ".png")
    cropped_image.save(new_path)