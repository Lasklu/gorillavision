from torch.utils.data.sampler import BatchSampler

class TripletBatchSampler(BatchSampler):
    def __init__(self, df, batch_size, drop_last=False):
        self.df = df
        self.batch_size = batch_size
        self.drop_last = drop_last
        # create one df for every class

    def __iter__(self):
        # randomly sample but ensure that at least one positive sample in batch
        pass
