import os
import shutil
import random
#This scripts takes defined datasets and splits them into train and test sets.
base_path_1 = "/scratch1/wildlife_conservation/data/gorillas_dante/face_cropped"
base_path = "/scratch1/all_gorillas"
output_path = os.path.join(base_path_1, 'eval_datasets', 'baseline_non_distinct')
bristol_dataset = os.path.join(base_path, 'bristol', 'face_images')
cxl_dataset = os.path.join(base_path, 'cxl', 'face_images_grouped')

classes_bristol = os.listdir(bristol_dataset)
random.shuffle(classes_bristol)
train_classes_bristol = classes_bristol[:5]

classes_cxl = os.listdir(cxl_dataset)
train_classes_cxl = random.sample(classes_cxl, round(len(classes_cxl) * 0.9))

datasets = [
	# {
	# 	"name": 'onlycxl_allperId_min20images_train_all_0.75',
	# 	'datasets': [
    #   	{
    #     'dataset_path': cxl_dataset,
	# 	'minimum_images_per_instance': 20,
	# 	'elements_per_id': 'all',
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 	"name": 'onlycxl_allperId_all_train_all_0.75',
	# 	'datasets': [
    #   	{
    #     'dataset_path': cxl_dataset,
	# 	'elements_per_id': 'all',
	# 	'minimum_images_per_instance': 0,
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 	"name": 'onlycxl_allperId_min6images_train_all_0.75',
	# 	'datasets': [
    #   	{
    #     'dataset_path': cxl_dataset,
	# 	'minimum_images_per_instance': 6,
	# 	'elements_per_id': 'all',
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 	"name": 'onlycxl_allperId_min10images_train_all_0.75',
	# 	'datasets': [
    #   	{
    #     'dataset_path': cxl_dataset,
	# 	'minimum_images_per_instance': 10,
	# 	'elements_per_id': 'all',
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 	"name": 'onlycxl_10perId_all_train_all_0.75',
	# 	'datasets': [
    #   	{
    #     'dataset_path': cxl_dataset,
	# 	'minimum_images_per_instance': 0,
	# 	'elements_per_id': 10,
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 	"name": 'onlybristol_allperId_all_train_all_0.75',
	# 	'datasets': [
    #   	{
    #     'dataset_path': bristol_dataset,
	# 	'elements_per_id': 'all',
	# 	'minimum_images_per_instance': 0,
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 	"name": 'onlybristol_10perId_all_train_all_0.75',
	# 	'datasets': [
    #   	{
    #     'dataset_path': bristol_dataset,
	# 	'elements_per_id': 10,
	# 	'minimum_images_per_instance': 0,
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },




	# {
	# 		"name": 'onlybristol_20perId_train_all_0.75',
	# 		'datasets': [
	# 		{'dataset_path': bristol_dataset,
	# 		'elements_per_id': 20,
	# 		'train_test_split': {
	# 			'train_classes': 'all',
	# 			'percentage': 0.75
	# 		}}]
	# 	},
	# {
	# 	"name": 'onlybristol_50perId_train_all_0.75',
	# 	'datasets': [
    #   	{'dataset_path': bristol_dataset,
	# 	'elements_per_id': 50,
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 	"name": 'onlybristol_AllperId_train_all_0.75',
	# 	'datasets': [
    #   	{'dataset_path': bristol_dataset,
	# 	'elements_per_id': 'all',
	# 	'train_test_split': {
	# 		'train_classes': 'all',
	# 		'percentage': 0.75
	# 	}}]
	# },#now two classes only for testing
	# {
	# 	"name": 'onlybristol_10perId_train_5_0.75',
	# 	'datasets': [
    #   	{'dataset_path': bristol_dataset,
	# 	'elements_per_id': 10,
	# 	'train_test_split': {
	# 		'train_classes': train_classes_bristol,
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 		"name": 'onlybristol_20perId_train_5_0.75',
	# 		'datasets': [
	# 		{'dataset_path': bristol_dataset,
	# 		'elements_per_id': 20,
	# 		'train_test_split': {
	# 			'train_classes': train_classes_bristol,
	# 			'percentage': 0.75
	# 		}}]
	# 	},
	# {
	# 	"name": 'onlybristol_50perId_train_5_0.75',
	# 	'datasets': [
    #   	{'dataset_path': bristol_dataset,
	# 	'elements_per_id': 50,
	# 	'train_test_split': {
	# 		'train_classes': train_classes_bristol,
	# 		'percentage': 0.75
	# 	}}]
	# },
	# {
	# 	"name": 'onlybristol_AllperId_train_5_0.75',
	# 	'datasets': [
    #   	{'dataset_path': bristol_dataset,
	# 	'elements_per_id': 'all',
	# 	'train_test_split': {
	# 		'train_classes': train_classes_bristol,
	# 		'percentage': 0.75
	# 	}}]
	# },
 
#  {#only cxl
# 		"name": 'onlycxl_AllperId_train_all_0.75',
# 		'datasets': [
#       	{'dataset_path': bristol_dataset,
# 		'elements_per_id': 'all',
# 		'train_test_split': {
# 			'train_classes': train_classes_cxl,
# 			'percentage': 0.75
# 		}}]
# 	},
#  	{
# 		"name": 'onlybristol_AllperId_train_90per_0.75',
# 		'datasets': [
#       	{'dataset_path': bristol_dataset,
# 		'elements_per_id': 'all',
# 		'train_test_split': {
# 			'train_classes': train_classes_bristol,
# 			'percentage': 0.75
# 		}}]
# 	},

# 	{
# 		"name": '10elementsperfolder_trainb1b2b3b4b5',
# 		'datasets': [
#       	{'dataset_path': bristol_dataset,
# 		'elements_per_id': 'all',
# 		'train_test_split': {
# 			'train_classes': train_classes_bristol,
# 			'percentage': 0.8
# 		}},
#         {
#         'dataset_path': cxl_dataset,
# 		'elements_per_id': 'all',
# 		'train_test_split': {
# 			'train_classes': train_classes_cxl,
# 			'percentage': 0.5
# 		}
# 		}]
# 	}#,
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
		identifier = f"{dataset['name']}"
		make_folder(identifier, output_path)
		make_folder(f"{identifier}/train", output_path)
		make_folder(f"{identifier}/test", output_path)
		for sub_dataset in dataset['datasets']:
			for id in os.listdir(sub_dataset['dataset_path']):
				if sub_dataset['train_test_split']['train_classes'] == 'all' or id in sub_dataset['train_test_split']['train_classes']:
					files = os.listdir(f"{sub_dataset['dataset_path']}/{id}")
					random.shuffle(files)
					if len(files) < sub_dataset["minimum_images_per_instance"]:
						print(f"WARNING: The class {id} fewer images than required by settings!")
						continue
					if round(len(files) * sub_dataset['train_test_split']['percentage']) >= len(files):
						print(f"WARNING: The class {id} has too few labels!")
						continue
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