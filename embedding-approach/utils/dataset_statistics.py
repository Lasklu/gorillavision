import os
def compute_statistics(train_path, test_path, type="bristol"):
    train_classes = os.listdir(train_path)
    test_classes = os.listdir(test_path)
    average_train = get_average_amount_of_images_per_label(train_path)
    average_test = get_average_amount_of_images_per_label(test_path)
    
    
    return {
		"amount_of_train_classes": len(train_classes),
		"amount_of_test_classes": len(test_classes),
		"average_amount_of_images_train": average_train,
		"average_amount_of_images_test": average_test,
		"type": type,
        "name": train_path.split("/")[-2]
	}

def get_average_amount_of_images_per_label(path):
    folders = os.listdir(path)
    amount_of_images = 0
    for folder in folders:
        if os.path.isdir(os.path.join(path, folder)):
            amount_of_images += len(os.listdir(os.path.join(path, folder)))
    return amount_of_images / len(folders)
    