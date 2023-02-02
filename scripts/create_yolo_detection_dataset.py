import os
import shutil


def make_folder(folder_path):
    if not os.path.exists(folder_path):
    	os.makedirs(folder_path)
def copy_files(src_folder, image_source, dst_folder, file_ending):
    for folder in os.listdir(src_folder):
        for file in os.listdir(os.path.join(src_folder, folder)):
            shutil.copy(
                		os.path.join(image_source, f"{file.split('.')[0]}.{file_ending}"),
                        os.path.join(dst_folder,folder, f"{file.split('.')[0]}.{file_ending}")
                        )
def do(src_path, main_path, image_source, type):
    isLabel = type == "labels"
    new_dir_data_path = os.path.join(main_path, type)
    new_dir_train_path = os.path.join(new_dir_data_path, "train")
    new_dir_test_path = os.path.join(new_dir_data_path, "test")
    new_dir_val_path = os.path.join(new_dir_data_path, "val")
    make_folder(new_dir_data_path)
    make_folder(new_dir_train_path)
    make_folder(new_dir_test_path)
    make_folder(new_dir_val_path)
    copy_files(src_folder=src_path, image_source=image_source, dst_folder=new_dir_train_path, file_ending="txt" if isLabel else "png")
    copy_files(src_folder=src_path, image_source=image_source, dst_folder=new_dir_val_path, file_ending="txt" if isLabel else "png")
    copy_files(src_folder=src_path, image_source=image_source, dst_folder=new_dir_test_path, file_ending="txt" if isLabel else "png")

def main():
	image_source = ""
	überpath = ""
	über_out_path = ""
	for dir in os.listdir(überpath):
		print(f"Doing for directory {dir}")
		new_dir_path = os.path.join(über_out_path, dir)
		make_folder(new_dir_path)
		do(os.path.join(überpath, dir), new_dir_path, image_source, "labels")
		do(os.path.join(überpath, dir), new_dir_path, image_source, "images")

if __name__ == '__main__':
    main()