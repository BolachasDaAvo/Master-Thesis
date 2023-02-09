import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as F
from PIL import Image


class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
            "shear_x": (torch.linspace(0.0, 0.3, self.num_bins), True),
            "shear_y": (torch.linspace(0.0, 0.3, self.num_bins), True),
            "translate_x": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], self.num_bins), True),
            "translate_y": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], self.num_bins), True),
            "rotate": (torch.linspace(0.0, 30.0, self.num_bins), True),
            "brightness": (torch.linspace(0.0, 0.9, self.num_bins), True),
            "contrast": (torch.linspace(0.0, 0.9, self.num_bins), True),
            "color": (torch.linspace(0.0, 0.9, self.num_bins), True),
            "sharpness": (torch.linspace(0.0, 0.9, self.num_bins), True),
            "posterize": (8 - (torch.arange(self.num_bins) / ((self.num_bins - 1) / 4)).round().int(), False),
            "solarize": (torch.linspace(255.0, 0.0, self.num_bins), False),
            "auto_contrast": (torch.tensor(0.0), False),
            "equalize": (torch.tensor(0.0), False),
            "perspective": (torch.linspace(0.0, 0.3, self.num_bins), False),
            "horizontal_flip": (torch.tensor(0.0), False),
            "vertical_flip": (torch.tensor(0.0), False),
            "gaussian_blur": (torch.linspace(0.0, 2.0, self.num_bins), False),
        }

    def forward(self, img, magnitude, name):
        op_meta = self._augmentation_space(F.get_image_size(img))
        magnitudes, signed = op_meta[name]
        magnitude = float(magnitudes[magnitude].item()) if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        img = self._apply_op(img, name, magnitude, interpolation=self.interpolation, fill=None)

        return img


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])


plt.rcParams["savefig.bbox"] = "tight"

transformer = Transform()
magnitudes = [1, 7, 13]
transforms = ["shear_x", "shear_y", "translate_x", "translate_y", "rotate", "brightness", "contrast", "color",
              "sharpness", "posterize", "solarize", "auto_contrast", "equalize", "perspective", "horizontal_flip",
              "vertical_flip", "gaussian_blur"]
orig_img = Image.open("").convert(
    mode="RGB")
save_root = ""

for t in transforms:
    transformed_imgs = [transformer.forward(orig_img, m, t) for m in magnitudes]
    plot(transformed_imgs)
    plt.savefig(f"{save_root}/{t}.png")

exit(0)

policies = [tfs.AutoAugmentPolicy.IMAGENET, tfs.AutoAugmentPolicy.CIFAR10, tfs.AutoAugmentPolicy.SVHN]
for policy in policies:
    transformed_imgs = [tfs.AutoAugment(policy, interpolation=tfs.InterpolationMode.BICUBIC).forward(orig_img) for _ in
                        range(3)]
    plot(transformed_imgs)
    plt.savefig(f"{save_root}/{policy}.png")

sequences = [1, 2, 3, 4]
magnitudes = [1, 7, 13]
for sequence in sequences:
    transformed_imgs = [tfs.RandAugment(sequence, m, 16, interpolation=tfs.InterpolationMode.BICUBIC).forward(orig_img)
                        for m in magnitudes]
    plot(transformed_imgs)
    plt.savefig(f"{save_root}/{sequence}.png")
