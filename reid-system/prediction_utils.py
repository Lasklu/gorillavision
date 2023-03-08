import cv2
import os

VIDEO_FORMATS = [".mp4", ".avi",]
IMAGE_FORMATS = [".png", ".jpg", ".jpeg"]
COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (0,128,0), (128,0,128), (245,222,179)]

def make_tracks(frame_results):
    tracks = {}

    for idx, frame_result in enumerate(frame_results):
        for result in frame_result:
            track_id = result[1]
            item = {
                "frame_idx": idx,
                "confidence": result[2],
                "bbox": result[3:6],
                "class": result[0],
            }
            if track_id not in tracks:
                tracks[track_id] = [item]
            else:
                tracks[track_id].append(item)

    return tracks

def join_tracks(tracks, identities):
    print(tracks.keys())
    print(identities)
    new_tracks = {}
    unknown_count = 0
    for track_id in tracks.keys():
        print(track_id)
        if track_id in identities:
            identity = identities[track_id]
            print(identity)
            if identity in new_tracks:
                # ugly validation for testint purposes
                for f in tracks[track_id]:
                    for f2 in new_tracks[identity]:
                        if f["frame_idx"] == f2["frame_idx"]:
                            print(f)
                            print(f2)
                            raise Exception("Duplicate individual detected")
                new_tracks[identity] += tracks[track_id]
            else:
                new_tracks[identity] = tracks[track_id]
        else:
            new_tracks[f"unknown-{unknown_count}"] = tracks[track_id]
            unknown_count += 1

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

    