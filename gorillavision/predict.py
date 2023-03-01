import os
import sys

import cv2
import numpy as np
from sklearn import neighbors
from PIL import Image

from detect import detect
from utils.plots import plot_one_box
from gorillavision.utils.image import transform_image
from gorillavision.model.triplet import TripletLoss

from  bridge_wrapper import *
from detection_helpers import *
from tracking_helpers import *
from prediction_utils import *

class GorillaVision():
    def __init___(self, file_paths, face_detection_model_path, body_detection_model_path, stageII_model_path, db_folder, out_folder):
        self.file_paths = file_paths
        self.out_folder = out_folder

        self.detector_face = Detector()
        self.detector_body = Detector()
        detector_face.load_model(face_detection_model_path)
        detector_body.load_model(body_detection_model_path)

        if contains_video(self.file_paths):
            self.tracker_face = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector_face)
            self.tracker_body = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector_body)
            self.video_cap = None

        self.init_identity_model(stageII_model_path, db_folder)
        self.identity_model_imgsz = (224, 224)

    def init_identity_model(self, model_path, db_folder):
        self.identity_model = TripletLoss.load_from_checkpoint(model_path)
        self.knn_classifier = neighbors.KNeighborsClassifier()
        db_embeddings = np.load(os.path.join(db_folder, "embeddings.npy"))
        db_labels = np.load(os.path.join(db_folder, "labels.npy"))
        self.knn_classifier.fit(db_embeddings, db_labels)

    def predict_all(self):
        for file_path in self.file_path:
            predict_video(file_path)
            """if is_video(file_path):
                predict_video(file_path)
            elif is_img(file_path):
                predict_img(file_path)
            else:
                raise Exception("Trying to predict unknown file type")"""
            
    def predict_identity(self, img):
        img = transform_image(img, self.identity_model_imgsz, "crop")
        with torch.no_grad():
            predicted_embedding = self.identity_model(img).numpy()
            predicted_id = self.knn_classifier.predict(predicted_embedding)
            return predicted_id

    def predict_img(self, img_path):
        pass

    def crop_to_bbox(self, frame_idx, bbox):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, unprocessed_img = video_cap.read()
        bbox = [int(v) for v in bbox]
        cropped_img = unprocessed_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return cropped_img

    def predict_video(self, video_path):
        self.video_cap = cv2.VideoCapture(video_path)
        frame_results_body = tracker_body.track_video(video_path, output=out_path, show_live = False, skip_frames = 0, count_objects = True, verbose=1)
        frame_results_face = tracker_face.track_video(video_path, output=out_path, show_live = False, skip_frames = 0, count_objects = True, verbose=1)
        body_tracks = make_tracks(frame_results_body)
        face_tracks = make_tracks(frame_results_face)

        identities = {}
        for track_key in face_tracks.keys():
            face_track = face_tracks[track_key]
            amount_to_select = min(len(tracks[track_id]), 5)
            selection = np.random.choice(tracks[track_id], amount_to_select, replace=False)
            imgs = [self.crop_to_bbox(res["frame_idx"], res["bbox"]) for res in selection]
            predictions = [self.predict_identity(img) for img in imgs]
            majority_element = max(set(predictions), key=predictions.count)
            identities[track_key] = majority_element

        face_tracks = join_tracks(face_tracks, identities)
        # body_tracks_identities = map_body_to_faceID(body_tracks, face_tracks)
        out_path = os.path.join(self.out_path, "out1.avi")
        self.save_as_video(body_tracks, body_tracks.keys()), out_path
        self.video_cap.release()

    def save_as_video(self, tracks, track_identities, out_path):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        vid_writer = cv2.VideoWriter(out_path, codec, fps, (w, h))

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
                    color = colors[i]
                    identity = track_identities[track_id]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len("Gorilla")+len(str(identity)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, "Gorilla" + " : " + str(identity),(int(bbox[0]), int(bbox[1]-11)),0, 1.2, (255,255,255),2, lineType=cv2.LINE_AA)    
            
            res = np.asarray(frame)
            res = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vid_writer.write(res)
            frame_count += 1


if __name__ == '__main__':
    files = ["/data/data/video1.MP4"]
    body_model_path = "/data/models/yolov7_body_model.pt"
    face_model_path = "/data/models/yolov7_face_model.pt"
    model2_path = "/data/models/identification_model.ckpt"
    db_folder = "/data/models/db"
    out_path = "/data/predictions"

    model = GorillaVision(files, body_model_path, face_model_path, db_folder, out_path)
    model.predict_all()
