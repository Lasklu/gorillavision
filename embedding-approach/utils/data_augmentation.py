from kornia.augmentation import  RandomRotation,  RandomHorizontalFlip, RandomPerspective, RandomPlanckianJitter, RandomBrightness, RandomErasing
import torch
import torch.nn as nn

class DataAugmentation(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.use_erase = config["use_erase"]
        self.use_geometric = config["use_geometric"]
        self.use_intensity = config["use_intensity"]

        self.geometric_transforms = nn.Sequential(
            # RandomRotation(degrees=360, p=0.3),
            RandomHorizontalFlip(p=0.3),
            RandomPerspective(p=0.3)
        )

        self.intensity_transforms = nn.Sequential(
            RandomPlanckianJitter(p=0.3),
            RandomBrightness(p=0.3),
        )

        self.erase_transforms = nn.Sequential(
            RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3), p=0.3)
        )

    @torch.no_grad()
    def forward(self, x):
        if self.use_erase:
            x = self.erase_transforms(x)
        if self.use_geometric:
            x = self.geometric_transforms(x)
        if self.use_intensity:
            x = self.intensity_transforms(x)
        return x