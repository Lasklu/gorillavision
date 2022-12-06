from csv import DictReader
import os

in_file = "/home/rohan/Downloads/all_metadata.csv"
out_folder = "./labels"

with open(in_file, 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    # iterate over each line as a ordered dictionary
    for row in csv_dict_reader:
        # row variable is a dictionary that represents a row in csv
        fn = row["image_id"]
        bbox = [row["voc_xmin"], row["voc_ymin"], row["voc_xmax"], row["voc_ymax"]]
        bbox = " ".join(bbox)
        if not os.path.exists(os.path.join(out_folder, fn + ".txt")):
            with open(os.path.join(out_folder, fn + ".txt"), 'w') as nf:
                nf.write("0 " + bbox + "\n")
        else:
            with open(os.path.join(out_folder, fn + ".txt"), 'a') as nf:
                nf.write("0 " + bbox + "\n")
