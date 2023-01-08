import copy
import random
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from numpy.random import default_rng

class BatchSamplerEnsurePositives(BatchSampler):
    def __init__(self, ds, batch_size, seed=123):
        self.ds = ds
        self.batch_size = batch_size
        self.rng = default_rng(seed=seed)
        self.classes_ds = {}
        self.labels = []
        # create one df for every class
        for idx, row in enumerate(DataLoader(ds)):
            self.labels.append(row["labels"].item())
            if row["labels"].item() not in self.classes:
                self.classes_ds[row["labels"].item()] = [idx]
            else: 
                self.classes_ds[row["labels"].item()].append(idx)

    def contains_positive_sample(self, batch):
        classes = {}
        for idx in batch:
            cls = self.labels[idx]
            if cls in classes:
                return True
            classes[cls] = 1
        return False

    def get_positive_sample(self, idx):
        cls = self.labels[idx]
        for i in self.classes_ds[cls]:
            if i != idx:
                return i
        raise Exception("Only one sample for this class existing")

    def __iter__(self):
        batch = [0] * self.batch_size
        idx_in_batch = 0
        idxes = range(0, len(self.ds))
        for idx in idxes:
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                if not self.contains_positive_sample(batch):
                    idx_anchor = batch[random.randint(0, len(batch) - 1)]
                    positive_idx = self.get_positive_sample_for(idx_anchor)
                    batch.pop()
                    batch.append(positive_idx)
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self) -> int:
        return (len(self.ds) + self.batch_size - 1) // self.batch_size 
