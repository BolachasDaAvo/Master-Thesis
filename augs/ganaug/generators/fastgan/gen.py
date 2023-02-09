import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from augs.ganaug.generators.fastgan.model import Generator


class ImageGenerator(torch.nn.Module):
    def __init__(self, config_paths, checkpoint_paths, device):
        super().__init__()

        self.device = device

        self._load_configs(config_paths)
        self._load_generators()
        self._load_checkpoints(checkpoint_paths)

        self.imgs = {i: [] for i in range(len(config_paths))}
        self.img_batch = 256
        self.batch_size = 4

    def _load_configs(self, config_paths):
        self.configs = {}
        for label, config_path in enumerate(config_paths):
            config = open(config_path, "r")
            self.configs[label] = json.load(config)

    def _load_generators(self):
        self.gens = {}
        for label, config in self.configs.items():
            self.gens[label] = Generator(
                image_size=config["image_size"],
                transparent=config["transparent"],
                greyscale=config["greyscale"],
                attn_res_layers=config["attn_res_layers"],
                freq_chan_attn=config["freq_chan_attn"],
            ).to(self.device)

    def _load_checkpoints(self, checkpoint_paths):
        for label, gen in self.gens.items():
            checkpoint_data = torch.load(checkpoint_paths[label], map_location=torch.device(self.device))["GAN"]
            filtered_checkpoint_data = OrderedDict()

            for k, v in checkpoint_data.items():
                if k.startswith("GE"):
                    filtered_checkpoint_data[k[3:]] = v

            gen.load_state_dict(filtered_checkpoint_data)
            gen.eval()

    def _visualize_imgs(self, imgs):
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.show()

    def _convert_imgs(self, imgs):
        imgs = imgs.detach()
        pil_imgs = []
        for img in imgs:
            img = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            pil_imgs.append(Image.fromarray(img))
        return pil_imgs

    def _get_img_batch(self, label):
        gen = self.gens[label]

        with torch.no_grad():
            for _ in range(self.img_batch // self.batch_size):
                fake_imgs = gen(torch.randn(self.batch_size, 256, device=self.device))
                fake_imgs = self._convert_imgs(fake_imgs)
                self.imgs[label].extend(fake_imgs)

    def forward(self, label):
        if len(self.imgs[label]) <= 0:
            self._get_img_batch(label)

        img = self.imgs[label].pop()
        return img
