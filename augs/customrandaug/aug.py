import math

import torch
import torchvision.transforms.functional as F
from torch import Tensor


class Augment(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M
        self.num_bins = 16
        self.interpolation = F.InterpolationMode.BICUBIC

    def _apply_op(self, img, op_name, magnitude, interpolation, fill):
        if op_name == "shear_x":
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(magnitude), 0.0],
                interpolation=interpolation,
                fill=fill,
            )
        elif op_name == "shear_y":
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(magnitude)],
                interpolation=interpolation,
                fill=fill,
            )
        elif op_name == "translate_x":
            img = F.affine(
                img,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif op_name == "translate_y":
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif op_name == "rotate":
            img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
        elif op_name == "brightness":
            img = F.adjust_brightness(img, 1.0 + magnitude)
        elif op_name == "color":
            img = F.adjust_saturation(img, 1.0 + magnitude)
        elif op_name == "contrast":
            img = F.adjust_contrast(img, 1.0 + magnitude)
        elif op_name == "sharpness":
            img = F.adjust_sharpness(img, 1.0 + magnitude)
        elif op_name == "posterize":
            img = F.posterize(img, int(magnitude))
        elif op_name == "solarize":
            img = F.solarize(img, magnitude)
        elif op_name == "auto_contrast":
            img = F.autocontrast(img)
        elif op_name == "equalize":
            img = F.equalize(img)
        elif op_name == "invert":
            img = F.invert(img)
        elif op_name == "identity":
            pass
        elif op_name == "perspective":
            (height, width) = F.get_image_size(img)
            half_height = height // 2
            half_width = width // 2
            topleft = [
                int(torch.randint(0, int(magnitude * half_width) + 1, size=(1,)).item()),
                int(torch.randint(0, int(magnitude * half_height) + 1, size=(1,)).item()),
            ]
            topright = [
                int(torch.randint(width - int(magnitude * half_width) - 1, width, size=(1,)).item()),
                int(torch.randint(0, int(magnitude * half_height) + 1, size=(1,)).item()),
            ]
            botright = [
                int(torch.randint(width - int(magnitude * half_width) - 1, width, size=(1,)).item()),
                int(torch.randint(height - int(magnitude * half_height) - 1, height, size=(1,)).item()),
            ]
            botleft = [
                int(torch.randint(0, int(magnitude * half_width) + 1, size=(1,)).item()),
                int(torch.randint(height - int(magnitude * half_height) - 1, height, size=(1,)).item()),
            ]
            startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            endpoints = [topleft, topright, botright, botleft]
            img = F.perspective(img, startpoints=startpoints, endpoints=endpoints, interpolation=interpolation,
                                fill=fill)
        elif op_name == "horizontal_flip":
            img = F.hflip(img)
        elif op_name == "vertical_flip":
            img = F.vflip(img)
        elif op_name == "gaussian_blur":
            img = F.gaussian_blur(img, kernel_size=[5, 5], sigma=magnitude)
        else:
            raise ValueError(f"The provided operator {op_name} is not recognized.")
        return img

    def _augmentation_space(self, image_size):
        return {
            "identity": (torch.tensor(0.0), False),
            "shear_x": (torch.linspace(0.0, 0.3, self.num_bins), True),
            "shear_y": (torch.linspace(0.0, 0.3, self.num_bins), True),
            "translate_x": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], self.num_bins), True),
            "translate_y": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], self.num_bins), True),
            "rotate": (torch.linspace(0.0, 30.0, self.num_bins), True),
            "perspective": (torch.linspace(0.0, 0.3, self.num_bins), False),
            "horizontal_flip": (torch.tensor(0.0), False),
            "vertical_flip": (torch.tensor(0.0), False),
            "contrast": (torch.linspace(0.0, 0.9, self.num_bins), True),
            "brightness": (torch.linspace(0.0, 0.9, self.num_bins), True),
            "auto_contrast": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        for _ in range(self.N):
            op_meta = self._augmentation_space(F.get_image_size(img))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.M].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = self._apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=None)

        return img
