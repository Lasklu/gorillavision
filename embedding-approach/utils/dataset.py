from utils.transformations import FillSizePad
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from typing import Tuple

class IndividualsDS(Dataset):

    def __init__(self, dataFrame, img_size: Tuple[int, int], img_preprocess):
        self.dataFrame = dataFrame
        
        transformations_crop = Compose([
            ToPILImage(),
            Resize(img_size),
            ToTensor(),
        ])

        transformations_pad = Compose([
            ToPILImage(),
            FillSizePad(img_size),
            ToTensor(),
        ])

        self.transformations = transformations_crop if img_preprocess == "crop" else transformations_pad
    
    def __getitem__(self, key):
        row = self.dataFrame.iloc[key]
        return {
            'images': self.transformations(row['images']),
            'labels': tensor([row['labels_numeric']], dtype=long),
        }
    
    def __len__(self):
        return len(self.dataFrame.index)