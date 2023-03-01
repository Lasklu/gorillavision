import os
import random
import shutil
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

class DatasetGenerator():
    def __init__(self, base_folder, dataset_configs, seed):
        self.base_folder = base_folder
        self.out_folder = os.path.join(base_folder, "new_splits")
        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)
        self.dataset_configs = dataset_configs
        self.seed=seed
        random.seed(seed)

    def make_splits(self):
        for ds_config in dataset_configs:
            df = self.make_initial_ds(ds_config["path"])
            df = df.sample(frac=1).reset_index(drop=True)

            if "min_images" in ds_config  and ds_config["min_images"] != None:
                print("reducing")
                df = self.reduce_to_min_img(df, ds_config["min_images"])
            
            if ds_config["open_set"]:
                splits = self.split_kfold_os(df, ds_config["k_fold"], ds_config["amount_eval"])
            else:
                splits = self.split_kfold_cs(df, ds_config["k_fold"])

            out_folder_name = f"{ds_config['name']}-openset={str(ds_config['open_set'])}"
            self.save_sets(out_folder_name, splits)
    
    def make_initial_ds(self, path):
        ds_path = os.path.join(self.base_folder, path)
        ds = {'path': [], 'labels': []}
        for individual_folder in os.listdir(ds_path):
            individual_path = os.path.join(ds_path, individual_folder)
            for file_name in os.listdir(individual_path):
                ds["path"].append(os.path.join(individual_path, file_name))
                ds["labels"].append(individual_folder)

        return pd.DataFrame(ds)

    def save_df(self, path, ds_type, df):
        path = os.path.join(path, ds_type)
        if not os.path.exists(path):
            os.mkdir(path)
        for index, row in df.iterrows():
            src_path = row['path']
            individual_folder = src_path.split(os.sep)[-2]
            new_individual_path = os.path.join(path, individual_folder)
            if not os.path.exists(new_individual_path):
                os.mkdir(new_individual_path)
            file_name = src_path.split(os.sep)[-1]
            target_path = os.path.join(path, individual_folder, file_name)
            shutil.copy(src_path, target_path)


    def save_sets(self, name, splits):
        for index, split in enumerate(splits):
            df_train, df_db, df_eval = split
            path = os.path.join(self.out_folder,f"{name}_{index}")
            if not os.path.exists(path):
                os.mkdir(path)
            self.save_df(path, "train", df_train)
            self.save_df(path, "database_set", df_db)
            self.save_df(path, "eval", df_eval)

    def split_kfold_cs(self, df, k):
        skf = StratifiedKFold(k, shuffle=False)
        data_splits = []
        values = df['path'].values
        labels = df['labels'].values
        for fold, (train_index, test_index) in enumerate(skf.split(values, labels)):
            train_values, test_values = values[train_index], values[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            df_train = pd.DataFrame({'path': train_values, 'labels': train_labels})
            df_eval = pd.DataFrame({'path': test_values, 'labels': test_labels})
            df_db = df_train.copy()

            print("db", df_db['labels'].value_counts())
            print("eval", df_eval['labels'].value_counts())
            print("train", df_train["labels"].value_counts())

            for path in df['path'].values:
                if path not in df_train['path'].values and path not in df_db['path'].values and path not in df_eval['path'].values:
                    raise Exception("Sanity check failed 0")
                if path in df_train['path'].values and path not in df_db['path'].values:
                    raise Exception("Sanit check failed 0.1")

            if len(df_train) != len(df_train[df_train['path'].isin(df_db['path'])]):
                raise Exception("Sanity check failed 1")

            if len(df_db[df_db['path'].isin(df_eval['path'])]) != 0:
                raise Exception("Sanity check failed 2")
            if len(df_train[df_train['path'].isin(df_eval['path'])]) != 0:
                raise Exception("Sanity check failed 2.1")

            data_splits.append([df_train, df_db, df_eval])
        return data_splits

    def reduce_to_min_img(self,df, min_img):
        return df[df.groupby('labels')['labels'].transform('size') > min_img]
        
    def split_kfold_os(self, df, k, amount_eval):
        all_classes = df['labels'].unique()
        random.shuffle(all_classes)

        kf = KFold(k, shuffle=False)
        data_splits = []
        for fold in kf.split(all_classes):
            train_classes, test_classes = fold
            train_classes = [all_classes[idx] for idx in train_classes] 
            test_classes = [all_classes[idx] for idx in test_classes]
            df_train = df[df['labels'].isin(train_classes)]
            df_db = df[df['labels'].isin(train_classes)]

            df_test = df[df['labels'].isin(test_classes)]
            df_eval = df_test.groupby('labels').apply(lambda x: x.sample(frac=amount_eval, random_state=self.seed))
            df_db_new = df_test[~df_test['path'].isin(df_eval['path'])]        
            df_db = df_db.append(df_db_new, ignore_index=True)

            # sanity checks
            cls_in_train = df_train['labels'].unique()
            cls_in_db = df_db['labels'].unique()
            cls_in_eval = df_eval['labels'].unique()

            for c in all_classes:
                if c not in cls_in_train and c not in cls_in_eval and c not in cls_in_db:
                    raise Exception("Sanity check failed 0")
                if c not in cls_in_db:
                    raise Exception("Sanit check failed 0.1")


            for ct in cls_in_train:
                if ct in cls_in_eval:
                    raise Exception("Sanity check failed 1")
                if ct not in cls_in_db:
                    raise Exception("Sanity check failed 1.1")

            if len(df_db[df_db['path'].isin(df_eval['path'])]) != 0:
                raise Exception("Sanity check failed 2")
            if len(df_train[df_train['path'].isin(df_eval['path'])]) != 0:
                raise Exception("Sanity check failed 2.1")

            data_splits.append([df_train, df_db, df_eval])
        return data_splits    

if __name__ == "__main__":
    base_folder = "/scratch1/wildlife_conservation/data/all_gorillas"
    dataset_configs = [
        {"path": "bristol/face_images", "name": "bristol_face", "k_fold": 4, "amount_eval": 0.3, "open_set": True},
        {"path": "cxl/face_images_grouped", "name": "cxl_face", "k_fold": 4, "amount_eval": 0.3, "open_set": True},
        {"path": "cxl/face_images_grouped", "name": "cxl_face_bigger6", "k_fold": 4, "amount_eval": 0.3, "min_images": 6, "open_set": True},
        {"path": "bristol/face_images", "name": "bristol_face", "k_fold": 4, "open_set": False},
        {"path": "cxl/face_images_grouped", "name": "cxl_face", "k_fold": 4, "open_set": False},
        {"path": "cxl/face_images_grouped", "name": "cxl_face_bigger6", "k_fold": 4, "min_images": 6, "open_set": False}
    ]
    DSGen = DatasetGenerator(base_folder, dataset_configs, 123)
    DSGen.make_splits()