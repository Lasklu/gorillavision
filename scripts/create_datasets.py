import os
import shutil
import random
#This scripts takes defined datasets and splits them into train and test sets.
base_path = "/scratch1/wildlife_conservation/data/gorillas_dante/face_cropped"
output_path = os.path.join(base_path, 'eval_datasets')
bristol_dataset = os.path.join(base_path, 'grouped_by_id_only_bristol')
cxl_dataset = os.path.join(base_path, 'grouped_by_id_only_dante')
datasets = [
	{
		"name": '10elementsperfolder_trainb1b2b3b4b5',
		'datasets': [
      	{'dataset_path': bristol_dataset,
		'elements_per_id': 'all',
		'train_test_split': {
			'train_classes': ['b1', 'b2', 'b3', 'b4', 'b5'],
			'percentage': 0.8
		}},
        {
        'dataset_path': cxl_dataset,
		'elements_per_id': 'all',
		'train_test_split': {
			'train_classes': ['folder1', 'folder2', 'folder3', 'folder4', 'folder5'],
			'percentage': 0.5
		}
		}]
	}#,
	#{
	# 	'dataset_path': "/Users/lukaslaskowski/Documents/HPI/9.Semester.nosync/Masterprojekt/eval/test_folder/bristol",
	# 	"name": '30elementsperfolder_trainb1b2',
	# 	'elements_per_id': 30,
	# 	'train_test_split': {
	# 		'train_classes': ['b1', 'b2'],
	# 		'percentage': 0.5
	# 	}
	# },
]
def main():
	for dataset in datasets:
		identifier = f"{dataset['name']}-split-{dataset['datasets'][0]['train_test_split']['percentage']}"
		make_folder(identifier, output_path)
		make_folder(f"{identifier}/train", output_path)
		make_folder(f"{identifier}/test", output_path)
		for sub_dataset in dataset['datasets']:
			for id in os.listdir(sub_dataset['dataset_path']):
				if id in sub_dataset['train_test_split']['train_classes']:
					print(id)
					print(os.listdir(f"{sub_dataset['dataset_path']}/{id}"))
					files = os.listdir(f"{sub_dataset['dataset_path']}/{id}")
					random.shuffle(files)
					print(files)
					file_counter = 0
					make_folder(id, f"{output_path}/{identifier}/train")
					make_folder(id, f"{output_path}/{identifier}/test")
					for file_name in files[:round(len(files) * sub_dataset['train_test_split']['percentage'])]:
						file_path = os.path.join(sub_dataset['dataset_path'], id, file_name)
						if sub_dataset['elements_per_id'] != 'all' and file_counter >= sub_dataset['elements_per_id']:
							target_path = os.path.join(output_path, identifier, 'test', id, file_name)
							shutil.copy(file_path, target_path)
							continue
						target_path = os.path.join(output_path, identifier, 'train', id, file_name)
						shutil.copy(file_path, target_path)
						file_counter += 1
					for file_name in files[round(len(files) * sub_dataset['train_test_split']['percentage']):]:
						file_path = os.path.join(sub_dataset['dataset_path'], id, file_name)
						target_path = os.path.join(output_path, identifier, 'test', id, file_name)
						shutil.copy(file_path, target_path)
				else:
					shutil.copytree(f"{sub_dataset['dataset_path']}/{id}", f"{output_path}/{identifier}/test/{id}")
   
def make_folder(folder_name, folder_path):
    folder = os.path.join(folder_path, folder_name)
    if not os.path.exists(folder):
    	os.makedirs(folder)
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()