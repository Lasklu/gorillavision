import copy
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from numpy.random import default_rng

class TripletBatchSampler(BatchSampler):
    # Not reccomended to use
    def __init__(self, ds, batch_size, seed=123):
        self.ds = ds
        self.batch_size = batch_size
        self.rng = default_rng(seed=seed)
        self.classes = {}
        # create one df for every class
        for idx, row in enumerate(DataLoader(ds)):
            if row["labels"].item() not in self.classes:
                self.classes[row["labels"].item()] = [idx]
            else: 
                self.classes[row["labels"].item()].append(idx)


    def __iter__(self):
        batch = [0] * self.batch_size
        idx_in_batch = 0
        cur_classes = copy.deepcopy(self.classes)
        i = 0
        idx_seen = []
        # continue until each el of ds is added to some batch
        while i < len(self.ds):
            classes_seen = {}
            single_items = []
            while idx_in_batch < self.batch_size - 1:
                # all items already selected, since ds ws not splittable into equal batches
                if len(cur_classes.items()) == 0:
                    break
                # randomly select a class from which to select items
                selected_class = self.rng.choice(list(cur_classes.keys()),1)[0]
                class_vals = cur_classes[selected_class]
                # if we haven't chosen an item from this class yet we want to choose 2, so that we also have one positive sample
                if selected_class not in classes_seen:
                    if len(class_vals) < 2:
                        # if there aren't two available and we don't have a positive sample yet, we do not yet choose from this class
                        # we remove it from the list and add it to single items, to fill up later
                        single_items.append(cur_classes[selected_class][0])
                        cur_classes.pop(selected_class, None)
                        continue
                    # randomly choose two samples from this class and add to batch
                    selected_items = self.rng.choice(class_vals, 2, replace=False)
                    batch[idx_in_batch] = selected_items[0]
                    batch[idx_in_batch+1] = selected_items[1]
                    idx_seen.append(selected_items[0])
                    idx_seen.append(selected_items[1])
                    cur_classes[selected_class].remove(selected_items[0])
                    cur_classes[selected_class].remove(selected_items[1])
                    idx_in_batch += 2
                    i += 2
                    classes_seen[selected_class] = True
                else:
                    batch[idx_in_batch] = self.rng.choice(class_vals, 1)[0]
                    idx_seen.append(batch[idx_in_batch])
                    cur_classes[selected_class].remove(batch[idx_in_batch])
                    idx_in_batch += 1
                    i += 1
                if len(class_vals) == 0:
                    # if we just removed the last samples from this class, remove class so that we can break when all classes are empty
                    cur_classes.pop(selected_class, None)
            
            # if still space in batch, fill up with single items that can be used as negatives
            while len(single_items) > 0 and idx_in_batch < self.batch_size:
                batch[idx_in_batch] = single_items.pop()
                idx_seen.append(batch[idx_in_batch])
                idx_in_batch += 1
                i += 1

            yield batch
            idx_in_batch = 0
            batch = [0] * self.batch_size
            cur_classes = copy.deepcopy(self.classes)
        
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]
        

        print("Sanity Check - all samples used over all batches: ", i, len(self.ds), len(idx_seen) == len(set(idx_seen)))

    def __len__(self) -> int:
        return (len(self.ds) + self.batch_size - 1) // self.batch_size 
