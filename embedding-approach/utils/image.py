from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import torch
def transform_image(img, img_size):
    transformations = Compose([
            ToPILImage(),
            Resize(img_size),
            ToTensor(),
        ])
    img =  transformations(img)
    return torch.unsqueeze(img, dim=0) #remove this when having multiple images