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
    if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
        return True
    return False

label_id = "" # descripes which label should be read. Currently there must be only one label of this type
images_folder = "/home/rohan/Documents/Uni/Sem3/AI/gorilla-reidentification/scripts/images"
lables_folder = "/home/rohan/Documents/Uni/Sem3/AI/gorilla-reidentification/scripts/labels"
output_folder = os.path.join(lables_folder, "primary")

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


for file in os.listdir(images_folder):
    window = tkinter.Tk(className="bla")    
    file_name, file_extension = os.path.splitext(file)
    label_path = os.path.join(lables_folder, file_name + ".txt")
    if file_extension != ".png" or not os.path.exists(label_path):
        continue

    image = Image.open(os.path.join(images_folder, file))
    bboxes_yolo = read_labels(label_path)
    bboxes = [yolobbox2bbox(*bbox, image.width, image.height) for bbox in bboxes_yolo]
    if len(bboxes) == 0:
        continue

    image = Image.open(os.path.join(images_folder, file))
    canvas = tkinter.Canvas(window, width=image.size[0], height=image.size[1])
    canvas.pack()
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)
    for bbox in bboxes:
        canvas.create_rectangle(*bbox, outline="#05f", width=1)

    def callback(event):
        for bbox in bboxes:
            if in_bbox(event.x, event.y, bbox):
                new_label_path = os.path.join(output_folder, file_name + ".txt")
                with open(new_label_path, 'w') as nf:
                    bbox_yolo = bboxes_yolo[bboxes.index(bbox)]
                    nf.write("0 " + " ".join([str(val) for val in bbox_yolo]))
                    window.destroy()
                return
        print("Unexpected behaviour: Clicked and not in bounding box")

    def skip():
        window.destroy()

    canvas.bind("<Button-1>", callback)
    canvas.bind('<Return>', skip)
    tkinter.mainloop()
    