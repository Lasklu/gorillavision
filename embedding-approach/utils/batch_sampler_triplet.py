import copy
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from numpy.random import default_rng

class TripletBatchSampler(BatchSampler):
    def __init__(self, ds, batch_size, drop_last=False, seed=123):
        self.ds = ds
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = default_rng(seed=seed)
        self.classes = {}
        # create one df for every class
        for idx, (x, y) in enumerate(DataLoader(ds)):
            if y not in self.classes:
                self.classes[y] = [idx]
            else: 
                self.classes[y].append(idx)


    def __iter__(self):
        batch = [0] * self.batch_size
        idx_in_batch = 0
        cur_classes = copy.deepcopy(self.classes)
        for i in range(0, len(self.ds)):
            classes_seen = {}
            while idx_in_batch < self.batch_size:
                if len(cur_classes.items()) == 0:
                    break
                selected_class = self.rng.choice(cur_classes.keys(),1)
                class_vals = cur_classes[selected_class]
                if selected_class not in classes_seen:
                    if len(class_vals) < 2:
                        # remove this class if cant use?
                        continue
                    selected_items = self.rng.choice(class_vals, 2)
                    batch[idx_in_batch] = selected_items[0]
                    batch[idx_in_batch+1] = selected_items[1]
                    class_vals.remove(selected_items[0])
                    class_vals.remove(selected_items[1])
                    idx_in_batch += 2
                else:
                    batch[idx_in_batch] = self.rng.choice(class_vals, 1)
                if len(class_vals) == 0:
                    cur_classes.pop(selected_class, None)
            
            yield batch
            idx_in_batch = 0
            batch = [0] * self.batch_size
            cur_classes = copy.deepcopy(self.classes)

        if idx_in_batch > 0:
            yield batch[:idx_in_batch]
