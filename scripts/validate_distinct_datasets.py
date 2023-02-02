import os

ds_path = "/scratch1/gorilla_experiment_splits/train_test_distinct"

for ds in os.listdir(ds_path):
    ds_base_path = os.path.join(ds_path, ds)
    train_individuals = [ind for ind in os.listdir(os.path.join(ds_base_path, "train"))]
    db_individuals = [ind for ind in os.listdir(os.path.join(ds_base_path, "database_set"))]
    eval_individuals = [ind for ind in os.listdir(os.path.join(ds_base_path, "eval"))]

    if any(item in train_individuals for item in db_individuals):
        raise Exception("individual from db in train")
    if any(item in train_individuals for item in eval_individuals):
        raise Exception("individual from eval in train")
    
    for ind in os.listdir(os.path.join(ds_base_path, "eval")):
        for f in os.listdir(os.path.join(ds_base_path, "eval", ind)):
            if f in list(os.listdir(os.path.join(ds_base_path, "database_set", ind))):
                raise Exception("Eval image in db already")

    