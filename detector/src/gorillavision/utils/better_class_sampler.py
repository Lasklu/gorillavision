import copy
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from numpy.random import shuffle, choice
import wandb

class BatchSamplerByClass(BatchSampler):
    def __init__(self, ds, seed=123, classes_per_batch=15, samples_per_class=3):
        # Uses every class once per batch. For every class takes min(smaples_per_class, len(class.samples))
        
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
        self.samples_per_class = samples_per_class
        self.batch_size = self.samples_per_class * self.classes_per_batch
        np.random.seed(seed)

    def __iter__(self):
        current_classes = list(self.classes_ds.keys())
        for i in range(0, self.__len__()):
            batch = [0] * self.batch_size
            idx_in_batch = 0
            amount_cls = min(self.classes_per_batch, len(current_classes))
            classes = np.random.choice(current_classes, amount_cls, replace=False)
            current_classes = [c for c in current_classes if c not in classes]
            for i in range(0, len(classes)):
                num_samples = min(self.samples_per_class, len(self.classes_ds[classes[i]]))
                selected_idx = np.random.choice(self.classes_ds[classes[i]], num_samples, replace=False)
                batch[idx_in_batch:idx_in_batch + len(selected_idx)] = selected_idx
                idx_in_batch += len(selected_idx)
            yield batch

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size 
