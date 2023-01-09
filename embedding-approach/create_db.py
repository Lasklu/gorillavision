import os
import csv
import cv2

def main():
    image_folder = ""
    db_file = ""
    pretrained_model_path = ""
    
    if not os.path.exists(db_file):
        with open(db_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["label", "embedding"])

    model = TripletLoss.load_from_checkpoint(pretrained_model_path)

    with open(db_file, 'a') as db:
        db_writer = csv.writer(db)
        for img_file in os.listdir(image_folder):
            label, ext = os.path.splitext(img_file)
            if ext not in [".png", ".jpg", ".jpeg"]:
                continue
            with torch.no_grad():
                img_path = os.path.join(image_folder, img_file)
                db_writer.writerow([label, model(cv2.imread(img_path))])

if __name__ == '__main__':
    main()