import torch
import torchvision.transforms as tfs


class Augment(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy_map = {
            "imagenet": tfs.AutoAugmentPolicy.IMAGENET,
            "cifar10": tfs.AutoAugmentPolicy.CIFAR10,
            "svhn": tfs.AutoAugmentPolicy.SVHN,
        }

        self.policy = self.policy_map[policy]

    def forward(self, img):
        return tfs.Compose([tfs.AutoAugment(self.policy, interpolation=tfs.InterpolationMode.BICUBIC)])(img)
