import os
import shutil
import random

#This scripts takes defined datasets and splits them into train,test and eeval sets.
base_path = "/scratch1/wildlife_conservation/data/all_gorillas"
output_path = os.path.join("/scratch1/wildlife_conservation/data/gorilla_experiment_splits/exp5/")
bristol_dataset_faces = os.path.join(base_path, 'bristol', 'face_images')
cxl_dataset_faces = os.path.join(base_path, 'cxl', 'face_images_grouped')

datasets = [
	{
		"name": 'cxl-bigger6_final6',
		'datasets': [
			{'dataset_path': cxl_dataset_faces,
			'min_elements_per_id': 6,
			'train_test_split': {
				'distinct': True,
				'percentage': 0.7
			}}
		]
	},

]

def copy_to(files, id, folder, ds_path, ds_identifier):
	for file_name in files:
		file_path = os.path.join(ds_path, id, file_name)
		if not os.path.exists(os.path.join(output_path, ds_identifier, folder, id)):
			os.mkdir(os.path.join(output_path, ds_identifier, folder, id))
		target_path = os.path.join(output_path, ds_identifier, folder, id, file_name)
		shutil.copy(file_path, target_path)

def split_train_test_non_distinct(files, percentage, min_el, ds_path, ds_identifier, id, exact_elements_per_id):
	# if not enough images per for this individual, copy to test
	if min_el != None and not len(files[:round(len(files) * percentage)]) >= min_el:
		return

	if exact_elements_per_id != None:
		copy_to(files[:exact_elements_per_id], id, "train", ds_path, ds_identifier)
		copy_to(files[:exact_elements_per_id], id, "database_set", ds_path, ds_identifier)
		if not os.path.exists(os.path.join(output_path, ds_identifier, 'eval', id)):
			os.mkdir(os.path.join(output_path, ds_identifier, 'eval', id))
		copy_to(files[exact_elements_per_id:], id, "eval", ds_path, ds_identifier)
	else:
		copy_to(files[:round(len(files) * percentage)], id, "train", ds_path, ds_identifier)
		copy_to(files[:round(len(files) * percentage)], id, "database_set", ds_path, ds_identifier)
		if not os.path.exists(os.path.join(output_path, ds_identifier, 'eval', id)):
			os.mkdir(os.path.join(output_path, ds_identifier, 'eval', id))
		copy_to(files[round(len(files) * percentage):], id, "eval", ds_path, ds_identifier)

def split_train_test_distinct(files, percentage, min_el, ds_path, ds_identifier, id, train_classes):
	if min_el != None and not len(files) >= min_el:
		copy_to(files, id, "database_set", ds_path, ds_identifier)
		return
	
	if id in train_classes:
		copy_to(files, id, "train", ds_path, ds_identifier)
		copy_to(files, id, "database_set", ds_path, ds_identifier)
	else:
		# inverse - percentage = amount in eval, rest in db
		db_amount = int(len(files) * percentage)
		copy_to(files[db_amount:], id, "eval", ds_path, ds_identifier)
		copy_to(files[:db_amount], id, "database_set", ds_path, ds_identifier)

		#copy_to(files[:round(len(files) * percentage)], id, "eval", ds_path, ds_identifier)
		#copy_to(files[round(len(files) * percentage):], id, "database_set", ds_path, ds_identifier)

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
		make_folder(f"{identifier}/database_set", output_path)
		make_folder(f"{identifier}/eval", output_path)
		for sub_dataset in dataset['datasets']:
			# only used if distinct train_test split mode
			train_classes, test_classes = get_distinct_classes_split(sub_dataset['dataset_path'], sub_dataset['train_test_split']['percentage'])
			
			exact_elements_per_id = sub_dataset["exact_elements_per_id"] if "exact_elements_per_id" in sub_dataset else None
			if exact_elements_per_id:
				temp_path = os.path.join(output_path, "temp")
				if not os.path.exists(temp_path):
					os.mkdir(temp_path)
					make_reduced_ds(sub_dataset, temp_path)
					sub_dataset['dataset_path'] = temp_path

			for id in os.listdir(sub_dataset['dataset_path']):
				files = os.listdir(f"{sub_dataset['dataset_path']}/{id}")
				random.shuffle(files)
				if not sub_dataset["train_test_split"]["distinct"]:
					split_train_test_non_distinct(files, sub_dataset['train_test_split']['percentage'],
												  sub_dataset["min_elements_per_id"], sub_dataset['dataset_path'], identifier, id, exact_elements_per_id)
				else:
					split_train_test_distinct(files, sub_dataset['train_test_split']['percentage'],
											  sub_dataset["min_elements_per_id"], sub_dataset['dataset_path'], identifier, id, train_classes)
			
			if "exact_elements_per_id" in sub_dataset:
				shutil.rmtree(os.path.join(output_path, "temp"))

   
def make_folder(folder_name, folder_path):
    folder = os.path.join(folder_path, folder_name)
    if not os.path.exists(folder):
    	os.makedirs(folder)
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()