from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

class IndividualsDS(Dataset):

    def __init__(self, dataFrame, img_size):
        self.dataFrame = dataFrame
        
        self.transformations = Compose([
            ToPILImage(),
            Resize(img_size),
            ToTensor(), # [0, 1]
        ])
    
    def __getitem__(self, key):
        row = self.dataFrame.iloc[key]
        return {
            'image': self.transformations(row['image']),
            'label': tensor([row['label']], dtype=long),
        }
    
    def __len__(self):
        return len(self.dataFrame.index)