import os
import shutil
import random

#This scripts takes defined datasets and splits them into train,test and eval sets.
base_path = "/scratch1/all_gorillas"
output_path = os.path.join("/scratch1/gorilla_experiment_splits/train_test_non_distinct")
bristol_dataset_faces = os.path.join(base_path, 'bristol', 'face_images')
cxl_dataset_faces = os.path.join(base_path, 'cxl', 'face_images_grouped')


datasets = [
	{
		"name": 'bristol_all_0_75',
		'datasets': [
			{'dataset_path': bristol_dataset_faces,
			'min_elements_per_id': None,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}}
		]
	},
	{
		"name": 'bristol_10images_0_75',
		'datasets': [
			{'dataset_path': bristol_dataset_faces,
			'min_elements_per_id': None,
			'exact_elements_per_id': 10,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}}
		]
	},
	{
		"name": 'cxl_all_0_75',
		'datasets': [
			{'dataset_path': cxl_dataset_faces,
			'min_elements_per_id': None,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}}
		]
	},
	{
		"name": 'cxl_greater6_0_75',
		'datasets': [
			{'dataset_path': cxl_dataset_faces,
			'min_elements_per_id': 6,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}}
		]
	},
	{
		"name": 'cxl-bristol_greater100_0_75',
		'datasets': [
			{'dataset_path': cxl_dataset_faces,
			'min_elements_per_id': 100,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}},
			{'dataset_path': bristol_dataset_faces,
			'min_elements_per_id': 100,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}}
		]
	},
	{
		"name": 'cxl-bristol_0_75',
		'datasets': [
			{'dataset_path': cxl_dataset_faces,
			'min_elements_per_id': None,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}},
			{'dataset_path': bristol_dataset_faces,
			'min_elements_per_id': None,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}}
		]
	},
		{
		"name": 'cxl-bristol_greater_6_0_75',
		'datasets': [
			{'dataset_path': cxl_dataset_faces,
			'min_elements_per_id': 6,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}},
			{'dataset_path': bristol_dataset_faces,
			'min_elements_per_id': 6,
			'train_test_split': {
				'distinct': False,
				'percentage': 0.75
			}},
		]
	},
]

def copy_to(files, id, folder, ds_path, ds_identifier):
	for file_name in files:
		file_path = os.path.join(ds_path, id, file_name)
		target_path = os.path.join(output_path, ds_identifier, folder, id, file_name)
		shutil.copy(file_path, target_path)

def split_train_test_non_distinct(files, percentage, min_el, ds_path, ds_identifier, id):
	# if not enough images per for this individual, copy to test
	if min_el != None and not len(files) >= min_el:
		copy_to(files, id, "test", ds_path, ds_identifier)
		return
	
	copy_to(files[:round(len(files) * percentage)], id, "train", ds_path, ds_identifier)
	copy_to(files[round(len(files) * percentage):], id, "test", ds_path, ds_identifier)

def split_train_test_distinct(files, percentage, min_el, ds_path, ds_identifier, id, train_classes):
	if min_el != None and not len(files) >= min_el:
		copy_to(files, id, "test", ds_path, ds_identifier)
		return
	
	if id in train_classes:
		copy_to(files, id, "train", ds_path, ds_identifier)
	else:
		copy_to(files, id, "test", ds_path, ds_identifier)

def get_distinct_classes_split(folders_path, percentage):
	all_classes = os.listdir(folders_path)
	random.shuffle(all_classes)
	train_classes = all_classes[:round(len(all_classes) * percentage)]
	test_classes = all_classes[round(len(all_classes) * percentage):]
	return train_classes, test_classes

def make_reduced_ds(dataset, out_path):
	for individual in os.listdir(dataset['dataset_path']):
		files = os.listdir(os.path.join(dataset['dataset_path'], individual))
		random.shuffle(files)
		os.mkdir(os.path.join(out_path, individual))
		for f in files[:dataset["exact_elements_per_id"]]:
			src_path = os.path.join(dataset['dataset_path'], individual, f)
			target_path = os.path.join(out_path, individual, f)
			shutil.copy(src_path, target_path)

def main():
	for dataset in datasets:
		identifier = f"{dataset['name']}"
		make_folder(identifier, output_path)
		make_folder(f"{identifier}/train", output_path)
		make_folder(f"{identifier}/test", output_path)
		make_folder(f"{identifier}/val", output_path)
		for sub_dataset in dataset['datasets']:
			# only used if distinct train_test split mode
			train_classes, test_classes = get_distinct_classes_split(sub_dataset['dataset_path'], sub_dataset['train_test_split']['percentage'])

			if "exact_elements_per_id" in sub_dataset:
				temp_path = os.path.join(output_path, "temp")
				if not os.path.exists(temp_path):
					os.mkdir(temp_path)
					make_reduced_ds(sub_dataset, temp_path)
					sub_dataset['dataset_path'] = temp_path

			for id in os.listdir(sub_dataset['dataset_path']):
				files = os.listdir(f"{sub_dataset['dataset_path']}/{id}")
				random.shuffle(files)
				make_folder(id, f"{output_path}/{identifier}/train")
				make_folder(id, f"{output_path}/{identifier}/test")
				if not sub_dataset["train_test_split"]["distinct"]:
					split_train_test_non_distinct(files, sub_dataset['train_test_split']['percentage'],
												  sub_dataset["min_elements_per_id"], sub_dataset['dataset_path'], identifier, id)
				else:
					split_train_test_distinct(files, sub_dataset['train_test_split']['percentage'],
											  sub_dataset["min_elements_per_id"], sub_dataset['dataset_path'], identifier, id, train_classes)
			
			# create validation set with part of the data from test
			for id in os.listdir(os.path.join(f"{output_path}/{identifier}", "test")):
				make_folder(id, f"{output_path}/{identifier}/val")
				files = os.listdir(os.path.join(f"{output_path}/{identifier}", "test", id))
				random.shuffle(files)
				for file_name in files[:round(len(files) * 0.2)]:
					file_path = os.path.join(f"{output_path}/{identifier}", "test", id, file_name)
					target_path = os.path.join(output_path, identifier, 'val', id, file_name)
					shutil.move(file_path, target_path)
			
			if "exact_elements_per_id" in sub_dataset:
				shutil.rmtree(os.path.join(output_path, "temp"))

   
def make_folder(folder_name, folder_path):
    folder = os.path.join(folder_path, folder_name)
    if not os.path.exists(folder):
    	os.makedirs(folder)
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()