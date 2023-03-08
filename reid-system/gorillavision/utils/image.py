from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Normalize
from gorillavision.utils.transformations import FillSizePad
import torch

def transform_image(img, img_size, img_preprocess):
    transformations_crop = Compose([
            ToPILImage(),
            Resize(img_size),
            ToTensor()
        ])

    transformations_pad = Compose([
        ToPILImage(),
        FillSizePad(img_size),
        ToTensor()
    ])

    transformations = transformations_crop if img_preprocess == "crop" else transformations_pad
    img =  transformations(img)
    return torch.unsqueeze(img, dim=0) #remove this when having multiple images