import os
import sys

import cv2
import numpy as np
from sklearn import neighbors
from PIL import Image

from utils.plots import plot_one_box
from gorillavision.utils.image import transform_image
from gorillavision.model.triplet import TripletLoss
from gorillavision.utils.logger import logger

from  bridge_wrapper import *
from detection_helpers import *
from tracking_helpers import *
from prediction_utils import *

class GorillaVision:

    def __init__(self, file_paths, face_detection_model_path, body_detection_model_path, stageII_model_path, db_folder, out_folder):
        logger.info("Loading models")
        self.file_paths = file_paths
        self.out_folder = out_folder

        self.detector_face = Detector()
        self.detector_body = Detector()
        self.detector_face.load_model(face_detection_model_path)
        self.detector_body.load_model(body_detection_model_path)

        if contains_video(self.file_paths):
            self.tracker_face = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=self.detector_face)
            self.tracker_body = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=self.detector_body)
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
        logger.info("Starting predictions")
        for file_path in self.file_paths:
            if is_video(file_path):
                self.predict_video(file_path)
            elif is_img(file_path):
                self.predict_img(file_path)
            else:
                raise Exception("Trying to predict unknown file type")
        logger.info(f"All predictions completed. You can find the results under {self.out_folder}")
            
    def predict_identity(self, img):
        img = transform_image(img, self.identity_model_imgsz, "crop")
        with torch.no_grad():
            predicted_embedding = self.identity_model(img).numpy()
            predicted_id = self.knn_classifier.predict(predicted_embedding)
            return predicted_id[0]

    def predict_img(self, img_path):
        logger.info(f"Predicting image: {img_path}")
        full_img = cv2.imread(img_path)
        detector_res = self.detector_face.detect(full_img, plot_bb = False)
        if len(detector_res) != 0:
            bboxes = detector_res[:,:4]
            scores = detector_res[:,4]
            classes = detector_res[:,-1]
            num_objects = bboxes.shape[0]
        else:
            bboxes = []
            scores = []
            classes = []
            num_objects = 0
        
        cropped_imgs = [self.crop_to_bbox(full_img, [int(v) for v in bbox]) for bbox in bboxes]
        results = [self.predict_identity(img) for img in cropped_imgs]

        for bbox, identity in zip(bboxes, results):
            draw_label(full_img, bbox, identity, (255,0,0))
        out_path = os.path.join(self.out_folder, os.path.basename(img_path))
        logger.info(f"Predictions completed. Saving results under: {out_path}")
        cv2.imwrite(out_path, full_img)

        
    def get_img_at_frame(self, frame_idx):
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = self.video_cap.read()
        return img

    def crop_to_bbox(self, img, bbox):
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return cropped_img

    def predict_video(self, video_path):
        logger.info(f"Predicting video: {video_path}")
        self.video_cap = cv2.VideoCapture(video_path)
        frame_results_body = self.tracker_body.track_video(video_path, output=None, show_live = False, skip_frames = 0, count_objects = True, verbose=1)
        frame_results_face = self.tracker_face.track_video(video_path, output=None, show_live = False, skip_frames = 0, count_objects = True, verbose=1)
        # TODO Important!! revisit trackbuilding: If looking at the output of the tracker and then looking at the output of the final result, there are tracks missing!
        body_tracks = make_tracks(frame_results_body)
        face_tracks = make_tracks(frame_results_face)

        identities = {}
        # Predict identity on top-5 bbox scores of frames in track
        for track_key in face_tracks.keys():
            face_track = face_tracks[track_key]
            amount_to_select = min(len(face_track), 10)
            sorted_track = sorted(face_track, key=lambda e: e["confidence"], reverse=True)
            selection = sorted_track[:5]
            imgs = [self.crop_to_bbox(self.get_img_at_frame(res["frame_idx"]), [int(v) for v in res["bbox"][0]]) for res in selection]
            predictions = [self.predict_identity(img) for img in imgs]
            majority_element = max(set(predictions), key=predictions.count)
            identities[track_key] = majority_element

        # face_tracks = join_tracks(face_tracks, identities) -> Not working since predictions not good enough
        # ToDo: map faces to the corresponding body, to have a nicer vizualization over the body tracks
        # body_tracks_identities = map_body_to_faceID(body_tracks, face_tracks)
        # self.save_as_video(body_tracks, out_path)
        out_path = os.path.join(self.out_folder, os.path.splitext(str(os.path.basename(video_path)))[0] + ".avi")
        logger.info(f"Predictions completed. Saving results under: {out_path}")
        self.save_as_video(face_tracks, identities, out_path)
        self.video_cap.release()

    def save_as_video(self, tracks, identitites, out_path):
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        w = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        vid_writer = cv2.VideoWriter(out_path, codec, fps, (w, h))

        frame_count = 0
        total_boxes = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            for i, track_id in enumerate(tracks.keys()):
                data = [data for data in tracks[track_id] if data["frame_idx"] == frame_count]
                if len(data) > 0:
                    data = data[0]
                    bbox = [int(v) for v in data["bbox"][0]]
                    draw_label(frame, bbox, identitites[track_id], COLORS[i%len(COLORS)])

            vid_writer.write(frame)
            frame_count += 1


if __name__ == '__main__':
    files = ["/data/demo/data/HU22.png", "/data/demo/data/RC01.png", "/data/demo/data/video1.MP4"]
    body_model_path = "/data/demo/models/yolov7_body_model.pt"
    face_model_path = "/data/demo/models/yolov7_face_model.pt"
    model2_path = "/data/demo/models/identification_model.ckpt"
    db_folder = "/data/demo/models/db"
    out_path = "/data/demo/predictions"

    model = GorillaVision(files, face_model_path, body_model_path, model2_path, db_folder, out_path)
    model.predict_all()