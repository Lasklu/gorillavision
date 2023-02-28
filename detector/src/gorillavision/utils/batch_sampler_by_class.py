import copy
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from numpy.random import shuffle, choice
import wandb

class BatchSamplerByClass(BatchSampler):
    def __init__(self, ds, seed=123, classes_per_batch=15, samples_per_class=3):
        # each bach will have the size of classes_per_batch * samples_per_class
        # However this approach does not use all samples per epoch
        self.ds = ds
        self.classes_ds = {}
        self.labels = []
        # create one df for every class
        for idx, row in enumerate(DataLoader(ds)):
            self.labels.append(row["labels"].item())
            if row["labels"].item() not in self.classes_ds:
                self.classes_ds[row["labels"].item()] = [idx]
            else: 
                self.classes_ds[row["labels"].item()].append(idx)
        self.classes_per_batch = min(classes_per_batch, len(list(self.classes_ds.keys())))
        self.samples_per_class = min(min([len(v) for v in self.classes_ds.values()]), samples_per_class)
        self.batch_size = self.samples_per_class * self.classes_per_batch
        print("batch size class sampler", self.batch_size)
        np.random.seed(seed)

    def __iter__(self):
        for i in range(0, self.__len__()):
            batch = [0] * self.batch_size
            idx_in_batch = 0
            classes = np.random.choice(list(self.classes_ds.keys()), self.classes_per_batch, replace=False)
            for i in range(0, len(classes)):
                selected_idx = np.random.choice(self.classes_ds[classes[i]], self.samples_per_class, replace=False)
                batch[idx_in_batch:idx_in_batch + len(selected_idx)] = selected_idx
                idx_in_batch += len(selected_idx)
            yield batch

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size 
