import cv2
import os

VIDEO_FORMATS = [".mp4", ".avi",]
IMAGE_FORMATS = [".png", ".jpg", ".jpeg"]

def make_tracks(frame_results):
    tracks = {}

    for idx, frame_result in enumerate(frame_results):
        for result in frame_result:
            track_id = result[1]
            item = {
                "frame_idx": idx,
                "bbox": result[2:5],
                "class": result[0],
            }
            if track_id not in tracks:
                tracks[track_id] = [item]
            else:
                tracks[track_id].append(item)

    return tracks

def join_tracks(tracks, identities):
    new_tracks = {}
    for track_key in identities.keys():
        identity = identities[track_key]
        if identity not in new_tracks:
            new_tracks[identity] = tracks[track_key]
        else:
            new_tracks[identity] += tracks[track_key]
    
    return new_tracks

def is_img(file_path):
    return os.path.splitext(file_path)[1].lower() in IMAGE_FORMATS

def is_video(file_path):
    return os.path.splitext(file_path)[1].lower() in VIDEO_FORMATS

def contains_video(file_paths):
    return any(is_video(file_path) for file_path in file_paths)

def draw_label(img, bbox, identity, color):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len("Gorilla")+len(str(identity)))*17, int(bbox[1])), color, -1)
    cv2.putText(img, "Gorilla" + " : " + str(identity),(int(bbox[0]), int(bbox[1]-11)),0, 1.2, (255,255,255),2, lineType=cv2.LINE_AA)    

    