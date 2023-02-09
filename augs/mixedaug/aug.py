import torch

import augs.customrandaug.aug as customrandaug
import augs.ganaug.aug as ganaug


class Augment(torch.nn.Module):
    def __init__(self, N, M, p, gan, config_paths, checkpoint_paths, device):
        super().__init__()
        self.p = p
        self.craug = customrandaug.Augment(N, M)
        self.gnaug = ganaug.Augment(1.0, gan, "identity", None, config_paths, checkpoint_paths, device)

    def next_gan(self):
        self.gnaug.next_gan()

    def forward(self, img, label):
        if torch.rand((1,)) < self.p:
            img = self.gnaug.forward(img, label)
        return self.craug.forward(img)
