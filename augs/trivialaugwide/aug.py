import torch
import torchvision.transforms as tfs


class Augment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img):
        return tfs.Compose([tfs.TrivialAugmentWide(16, interpolation=tfs.InterpolationMode.BICUBIC)])(img)
