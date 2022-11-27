import os
import tkinter
from PIL import Image, ImageTk

def read_labels(file_path):
    labels = []
    with open(file_path) as label_file:
        for line in label_file:
            vals = line.split()[1:]
            labels.append([float(val) for val in vals])
    return labels

def yolobbox2bbox(x,y,w,h, img_w, img_h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1*img_w, y1*img_h, x2*img_w, y2*img_h

def in_bbox(x,y, bbox):
    pass

label_id = "" # descripes which label should be read. Currently there must be only one label of this type
images_folder = ""
lables_folder = ""
output_folder = os.path.join(lables_folder, "primary")

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

window = tkinter.Tk(className="bla")    

for file in os.listdir(images_folder):
    file_name, file_extension = os.path.splitext(file)
    label_path = os.path.join(lables_folder, file_name + ".txt")
    if file_extension != ".png" or not os.path.exists(label_path):
        continue

    bboxes_yolo = read_labels(label_path)
    bboxes = [yolobbox2bbox(bbox) for bbox in bboxes_yolo]
    if len(bboxes) == 0:
        continue

    image = Image.open(os.path.join(images_folder, file))
    canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
    for bbox in bboxes:
        canvas.create_rectangle(*bbox, outline="#fb0")
    canvas.pack()
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)

    def callback(event):
        for bbox in bboxes:
            if in_bbox(event.x, event.y, bbox):
                new_label_path = os.path.join(output_folder, file_name + ".txt")
                with open(new_label_path, 'w') as nf:
                    bbox_yolo = bboxes_yolo[bboxes_yolo.index(bbox)]
                    nf.write("0 " + " ".join([str(val) for val in bbox_yolo]))
                return

    canvas.bind("<Button-1>", callback)
    