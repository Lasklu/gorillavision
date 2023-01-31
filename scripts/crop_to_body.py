import os
from PIL import Image
import cv2
import numpy as np

# given a set of bounding boxes of bodies for each picture and the face of the primary gorilla
# crop to the body of this primary gorilla

# ensure naming is consistent for matching files in each folder
labels_folder = "./data/bristol/body_labels"
full_images_folder = "./data/bristol/full_images"
cropped_faced_folder = "./data/bristol/face_images/5"
output_folder = "./data/bristol/body_images/5"
img_ext = ".jpg"

not_identifiable = []

def load_labels(file_path, file_name):
    labels = []
    with open(file_path) as label_file:
        for line in label_file:
            vals = line.split()[1:]
            coords_yolo = [float(val) for val in vals]
            img = Image.open(os.path.join(full_images_folder, file_name + img_ext))
            coords = yolobbox2bbox(*coords_yolo, img.width, img.height)
            labels.append(list(coords))
    return labels

def yolobbox2bbox(x,y,w,h, img_w, img_h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1*img_w, y1*img_h, x2*img_w, y2*img_h

def calc_overlap_area(x1, y1, w1, h1, x2, y2, w2, h2):
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1+w1, x2+w2)
    y_bottom = min(y1+h1, y2+h2)

    if x_right > x_left and y_bottom > y_top:
        overlap = (x_right - x_left) * (y_bottom - y_top)
    else:
        overlap = 0
    return overlap

def get_bbox_of_face(face_img, full_image):
    result = cv2.matchTemplate(face_img, full_image, cv2.TM_CCOEFF_NORMED)
    face_w, face_h = face_img.shape[:2]
    img_height, img_width, img_channels = full_image.shape

    _, _, _, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    return (x, y, face_w, face_h)

def find_primary_bb(body_bb_labels, full_image, face_image, file_name):
    bbox_of_face = get_bbox_of_face(face_image, full_image)
    highest_overlap = 0
    highest_overlap_bb = None
    for bb in body_bb_labels:
        overlap = calc_overlap_area(*bbox_of_face, *bb)
        if overlap > highest_overlap:
            highest_overlap = overlap
            highest_overlap_bb = bb

    w, h = face_image.shape[:2]
    area_of_face = w*h
    # validate that we actually found the correct gorilla for this
    if highest_overlap < 0.8* area_of_face:
        return None
    return highest_overlap_bb

for img_file in os.listdir(cropped_faced_folder):
    file_name, ext = os.path.splitext(img_file)
    label_file = f"{file_name}.txt"
    file_name, ext = os.path.splitext(label_file)
    if not os.path.exists(os.path.join(labels_folder, label_file)):
        continue
    body_labels = load_labels(os.path.join(labels_folder, label_file), file_name)
    full_image = cv2.imread(os.path.join(full_images_folder, f"{file_name}.jpg"))
    cropped_face_image = cv2.imread(os.path.join(cropped_faced_folder, img_file))
    primary_bb = find_primary_bb(body_labels, full_image, cropped_face_image, file_name)
    if primary_bb == None:
        continue
    image = Image.open(os.path.join(full_images_folder, f"{file_name}{img_ext}"))
    cropped_img = image.crop(tuple(primary_bb))
    out_path = os.path.join(output_folder, file_name + ".png")
    cropped_img.save(out_path)
