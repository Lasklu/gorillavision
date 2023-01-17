from kornia.augmentation import  RandomRotation,  RandomHorizontalFlip
import torch
import torch.nn as nn

class DataAugmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            RandomRotation(degrees=90, p=0.25),
            RandomHorizontalFlip(p=0.25),
        )

    @torch.no_grad()
    def forward(self, x):
        x_out = self.transforms(x)
        return x_out