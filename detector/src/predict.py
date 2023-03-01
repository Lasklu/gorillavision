import os
import sys

import cv2
import numpy as np
from sklearn import neighbors

from detect import detect
from utils.plots import plot_one_box
from gorillavision.utils.image import transform_image
from gorillavision.model.triplet import TripletLoss

def main(video_path, stageI_model_path, stageII_model_path, db_folder, out_path):

    # get bounding box and conf value for each frame. Format = [[class, confidence, bb1, bb2, bb3, bb4]]
    res_stageI = detect(video_path, stageI_model_path, "0", 640)
    res_stageI = np.array(res_stageI)

    # ToDo replace and implement tracks
    res_stageI = res_stageI[:, :, :1]

    video_cap = cv2.VideoCapture(video_path)

    # ToDo: Do this for each individual
    # get frame of predictions with highest confidence, to perform classification on
    max_prediction_index = res_stageI.argmax(axis=0)[1]
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, max_prediction_index)
    _, unprocessed_img = video_cap.read()
    bbox = res_stageI[max_prediction_index][2:6]
    bbox = [int(v) for v in bbox]
    #ToDo ensure bbox coordinates are correct
    cropped_img = unprocessed_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cv2.imwrite("/data/file1.png", cropped_img)

    # stage II
    """model = TripletLoss.load_from_checkpoint(model_path)
    knn_classifier = neighbors.KNeighborsClassifier()
    db_embeddings = np.load(os.path.join(db_folder, "embeddings.npy"))
    db_labels = np.load(os.path.join(db_folder, "labels.npy"))
    knn_classifier.fit(db_embeddings, db_labels)
    img = transform_image(img, (224, 224), "crop")

    predicted_id = "None"
    with torch.no_grad():
            predicted_embedding = model(img).numpy()
            predicted_id = knn_classifier.predict(predicted_embedding)"""

    predicted_id = "Lukas"
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame_res in res_stageI:
        xyxy = frame_res[2:6]
        xyxy = [int(v) for v in xyxy]
        ret, frame = video_cap.read()
        plot_one_box(xyxy, frame, label=predicted_id, color=(255,0,0), line_thickness=3)
        vid_writer.write(frame)

    video_cap.release()

if __name__ == '__main__':
    vid_path = "/data/video3.MP4"
    model1_path = "/gorilla-reidentification/detector/src/runs/best_weights/bristol_face_detection_best.pt"
    model2_path = ""
    db_folder = ""
    out_path = "/data/video_predicted.MP4"

    main(vid_path, model1_path, model2_path, db_folder, out_path)