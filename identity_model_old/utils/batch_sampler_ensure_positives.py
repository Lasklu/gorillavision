import copy
import random
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from numpy.random import shuffle, choice

class BatchSamplerEnsurePositives(BatchSampler):
    def __init__(self, ds, batch_size, seed=123):
        self.ds = ds
        self.batch_size = batch_size
        self.classes_ds = {}
        self.labels = []
        # create one df for every class
        for idx, row in enumerate(DataLoader(ds)):
            self.labels.append(row["labels"].item())
            if row["labels"].item() not in self.classes_ds:
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

    def get_positive_sample_for(self, idx):
        cls = self.labels[idx]
        for i in self.classes_ds[cls]:
            if i != idx:
                return i
        raise Exception("Only one sample for this class existing. Class_id: " + str(cls))

    def __iter__(self):
        batch = [0] * self.batch_size
        idx_in_batch = 0
        idxes = list(range(0, len(self.ds)))
        shuffle(idxes)
        possibilities_for_positive_anchor = [i for i in range(0, len(self.ds))]
        shuffle(possibilities_for_positive_anchor)

        def make_positive_pairs():
        # create a pair of (anchor, positive)
            while len(possibilities_for_positive_anchor) != 0:
                anchor_idx = possibilities_for_positive_anchor.pop()
                anchor_cls = self.labels[anchor_idx]
                if len(self.classes_ds[anchor_cls]) >= 2:
                    # choices are the available positives except the chosen anchor
                    positive_idx = None
                    while True:
                        rnd_int = random.randint(0, len(self.classes_ds[anchor_cls]) - 1)
                        positive_idx = self.classes_ds[anchor_cls][rnd_int]
                        if positive_idx != anchor_idx:
                            break
                    if positive_idx in idxes:
                        idxes.remove(positive_idx)
                    return (anchor_idx, positive_idx)
            raise Exception("Could not make positive pairs, sice all available classes left have less than 2 els")

        # create a positive pair for each batch, to ensure that each batch has at least one positive
        positive_pairs = [make_positive_pairs() for i in range(self.__len__())]
        count = 0
        while count < len(self.ds) - 2 * self.__len__():
            if idx_in_batch == 0:
                positive_pair = positive_pairs.pop()
                batch[0] = positive_pair[0]
                batch[1] = positive_pair[1]
                idx_in_batch += 2
            batch[idx_in_batch] = idxes[count]
            count += 1
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]
        count = 0

    def __len__(self) -> int:
        return (len(self.ds) + self.batch_size - 1) // self.batch_size 
