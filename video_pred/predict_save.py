import os
import pickle
import sys

import cv2
import numpy as np
from sklearn import neighbors

from detect import detect
from utils.plots import plot_one_box
from gorillavision.utils.image import transform_image
from gorillavision.model.triplet import TripletLoss


from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder

def main(video_path, stageI_model_path, stageII_model_path, db_folder, out_path):
    """detector = Detector()
    detector.load_model(stageI_model_path,)

    tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)
    frame_results = tracker.track_video(video_path, output=out_path, show_live = False, skip_frames = 0, count_objects = True, verbose=1)

    model = TripletLoss.load_from_checkpoint(model_path)
    knn_classifier = neighbors.KNeighborsClassifier()
    db_embeddings = np.load(os.path.join(db_folder, "embeddings.npy"))
    db_labels = np.load(os.path.join(db_folder, "labels.npy"))
    knn_classifier.fit(db_embeddings, db_labels)

    print(len(frame_results))
    tracks = {}
    for idx, frame_result in enumerate(frame_results):
        for result in frame_result:
            track_id = result[1]
            item = {
                "frame_idx": idx,
                "bbox": result[2:5],
                "class": result[0],
                "individual_id": None
            }
            if track_id not in tracks:
                tracks[track_id] = [item]
            else:
                tracks[track_id].append(item)

    for track in tracks.values():
        track = sorted(track, key=lambda d: d['frame_idx'])

    ids = ["kayana", "afia", "kukuenu", "enrald"]
    for track_id in tracks.keys():
        # ToDo: Choose 5 images in track that have highest detection score, for better results
        amount_to_select = min(len(tracks[track_id]), 5)
        selection = np.random.choice(tracks[track_id], amount_to_select, replace=False)

        predictions = []
        for item in selection:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, max_prediction_index)
            _, unprocessed_img = video_cap.read()
            bbox = item["bbox"]
            cropped_img = unprocessed_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite("/data/temp-face.png", cropped_img)

            img = transform_image(cropped_img, (224, 224), "crop")
            with torch.no_grad():
                predicted_embedding = model(img).numpy()
                predicted_id = knn_classifier.predict(predicted_embedding)
                predictions.append(predicted_id)

        majority_element = max(set(predictions), key = predictions.count)
    for e in tracks[track_id]:
        e["individual_id"] = f"{track_id}"

    pickle.dump(tracks, open('/data/tracks.pkl', 'wb'))"""

    tracks = pickle.load(open('/data/tracks.pkl', 'rb'))
    # ToDo join same tracks into one and give new name

    tracks[12] = [data for data in tracks[12] if data["frame_idx"] > 770]
    tracks[4] = tracks[4] + tracks[9]
    tracks[3] = tracks[3] + tracks[13]
    tracks[2] = tracks[2] + tracks[12]
    tracks[5] = tracks[5] + tracks[14]

    tracks.pop(9, None)
    tracks.pop(13, None)
    tracks.pop(12, None)
    tracks.pop(14, None)

    # 4=9
    # delete frist frames from 12
    # 3 = 13
    # 2 = 12
    # 14 = 5

    names = ["Afia", "Kayana", "Iriba", "Dikembe", "Laskluklu", "Biyomo", "Naira", "Kedar", "Mutombo", "Zimber", "Renua", "Santaro",]
    for track in tracks.values():
        individual_name = names.pop()
        for dic in track:
            dic["individual_id"] = individual_name

    print("starting output process")
    video_cap = cv2.VideoCapture(video_path)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"XVID")
    vid_writer = cv2.VideoWriter(out_path, codec, fps, (w, h))

    for track_id in tracks.keys():
        print(f"{track_id}----------------------")
        print([data["frame_idx"] for data in tracks[track_id]])

    frame_count = 0
    total_boxes = 0
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (0,128,0), (128,0,128), (245,222,179)]
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        for i, track_id in enumerate(tracks.keys()):
            data = [data for data in tracks[track_id] if data["frame_idx"] == frame_count]
            if len(data) > 0:
                data = data[0]
                bbox = [int(v) for v in data["bbox"][0]]
                total_boxes += 1
                color = colors[i%len(colors)]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-40)), (int(bbox[0])+(len("Gorilla")+len(str(data["individual_id"])))*21, int(bbox[1])), color, -1)
                cv2.putText(frame, "Gorilla" + " : " + str(data["individual_id"]),(int(bbox[0]), int(bbox[1]-11)),0, 1, (255,255,255),2, lineType=cv2.LINE_AA)    

                #plot_one_box(xyxy, frame, label=data["individual_id"], color=(255,0,0), line_thickness=3)
        
        res = np.asarray(frame)
        res = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vid_writer.write(res)
        frame_count += 1

    video_cap.release()

if __name__ == '__main__':
    vid_path = "/data/video3.MP4"
    model1_path = "/gorilla-reidentification/detector/src/runs/best_weights/yolov7_gorilla_dante_body_noid_fixed_1.pt"
    model2_path = ""
    db_folder = ""
    out_path = "/data/video_final_predicted.avi"

    main(vid_path, model1_path, model2_path, db_folder, out_path)