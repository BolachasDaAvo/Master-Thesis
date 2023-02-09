import torch

import augs.ganaug.selectors.identity as identity_selector
import augs.ganaug.selectors.realism as realism_selector
import augs.ganaug.selectors.varied as varied_selector
from augs.ganaug.generators.fastgan.gen import ImageGenerator
from augs.ganaug.generators.studiogan.gen import ImageGenerator


class Augment(torch.nn.Module):
    def __init__(self, p, gan, selector_name, data, config_path=None, checkpoint_path=None, device=None):
        super().__init__()
        self.p = p
        self.gan = gan
        self.selector_name = selector_name
        self.selector = None
        self.data = data
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.current = -1
        self.gen = None
        self.classifier_checkpoints = []

    def _get_generator(self, gan, config_path, checkpoint_path, selector, device):
        if gan in ["ACGAN", "ADCGAN", "ContraGAN", "MHGAN", "ReACGAN", "cStyleGAN2", "ProjGAN", "BigGAN"]:
            return ImageGenerator(config_path, checkpoint_path, selector, device)
        elif gan in ["FastGAN"]:
            return ImageGenerator(config_path, checkpoint_path, device)
        else:
            raise NotImplementedError("invalid gan type")

    def _get_selector(self, selector_name, real_images, device, model_path):
        if selector_name == "identity":
            selector = identity_selector.Selector()
        elif selector_name == "realism":
            selector = realism_selector.Selector(real_images, device)
        elif selector_name == "varied":
            selector = varied_selector.Selector(real_images, device, model_path)
        else:
            raise NotImplementedError("invalid selector type")

        return selector

    def next_gan(self):
        self.current = self.current + 1 if self.current + 1 < len(self.config_path) else 0

        real_images = None
        if self.data:
            real_images, _, _, _ = self.data.next_dataset()
            real_images = real_images.data

        # Free memory
        del self.selector
        del self.gen

        self.selector = self._get_selector(self.selector_name, real_images, self.device,
                                           self.classifier_checkpoints[self.current])
        self.gen = self._get_generator(self.gan, self.config_path[self.current], self.checkpoint_path[self.current],
                                       self.selector, self.device)

    def forward(self, img, label):
        if torch.rand((1,)) > self.p:
            return img

        return self.gen.forward(label)
