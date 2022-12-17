from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from typing import Tuple
class IndividualsDS(Dataset):

    def __init__(self, dataFrame, img_size: Tuple[int, int]):
        self.dataFrame = dataFrame
        
        self.transformations = Compose([
            ToPILImage(),
            Resize(img_size),
            ToTensor(),
        ])
    
    def __getitem__(self, key):
        row = self.dataFrame.iloc[key]
        return {
            'images': self.transformations(row['images']),
            'labels': tensor([row['labels_numeric']], dtype=long),
        }
    
    def __len__(self):
        return len(self.dataFrame.index)