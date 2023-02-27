import cv2
import numpy as np

from detector.detect import detect
from detector.utils.plots import plot_one_box
from utils.image import transform_image

def main(video_path, stageI_model_path, stageII_model_path, db_folder):

    res_stageI = detect("video_path", stageI_model_path, "0")
    video_cap = cv2.VideoCapture('video_path')

    max_prediction_index = res_stageI.argmax(axis=0)[1]
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, max_prediction_index)
    _, unprocessed_img = video_cap.read()
    bbox = res_stageI[max_prediction_index][2:6]
    cropped_img = unprocessed_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    model = TripletLoss.load_from_checkpoint(model_path)
    knn_classifier = neighbors.KNeighborsClassifier()
    db_embeddings = np.load(os.path.join(db_folder, "embeddings.npy"))
    db_labels = np.load(os.path.join(db_folder, "labels.npy"))
    knn_classifier.fit(db_embeddings, db_labels)
    img = transform_image(img, (224, 224), "crop")

    predicted_id = "None"
    with torch.no_grad():
            predicted_embedding = model(img).numpy()
            predicted_id = knn_classifier.predict(predicted_embedding)

    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame_res in res_stageI:
        xyxy = frame_res[0][2:6]
        frame = video_cap.read()
        plot_one_box(xyxy, frame, label=predicted_id, color=colors[int(cls)], line_thickness=1)



    cap.release()

if __name__ == '__main__':
    vid_path = "/data/video1.MP4"
    model1_path = "/gorilla-reidentification/embedding-approach/detector/runs/best_weights/bristol_face_detection_best.pt"
    model2_path = ""
    db_folder = ""

    main(vid_path, model1_path, model2_path, db_folder)