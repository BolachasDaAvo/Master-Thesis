import torch
import torchvision.transforms as tfs


class Augment(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M

    def forward(self, img):
        return tfs.Compose([tfs.RandAugment(self.N, self.M, 16, interpolation=tfs.InterpolationMode.BICUBIC)])(img)
