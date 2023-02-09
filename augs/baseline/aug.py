import torch


class Augment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img):
        return img
