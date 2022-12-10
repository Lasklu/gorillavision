from csv import DictReader
import os
from PIL import Image


in_file = "/home/rohan/Downloads/all_metadata.csv"
img_folder = "/home/rohan/Downloads/data"
out_folder = "./labels_good"

def voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

with open(in_file, 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    # iterate over each line as a ordered dictionary
    for row in csv_dict_reader:
        # row variable is a dictionary that represents a row in csv
        if row["bb_confirmed"] != "True":
            continue
        fn = row["image_id"]
        img_path = os.path.join(img_folder, fn + ".jpeg")
        if not os.path.exists(img_path):
            continue
       
        img = Image.open(img_path)
        bbox = [float(row["voc_xmin"]) * img.width, float(row["voc_ymin"]) * img.height, float(row["voc_xmax"]) * img.width, float(row["voc_ymax"]) * img.height]
        bbox = voc_to_yolo(*bbox, img.width, img.height)
        bbox = " ".join([str(val) for val in bbox])
        if not os.path.exists(os.path.join(out_folder, fn + ".txt")):
            with open(os.path.join(out_folder, fn + ".txt"), 'w') as nf:
                nf.write("0 " + bbox + "\n")
        else:
            with open(os.path.join(out_folder, fn + ".txt"), 'a') as nf:
                nf.write("0 " + bbox + "\n")
