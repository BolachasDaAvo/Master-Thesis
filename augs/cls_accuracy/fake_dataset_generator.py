import glob
import os
import random
import shutil
from collections import namedtuple

import numpy as np
import torch

from augs.ganaug import aug

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels_map = {"covid": 0, "lung_opacity": 1, "normal": 2, "viral_pneumonia": 3}

Checkpoint = namedtuple("Checkpoint", ["gan", "root"])
saves = {"ACGAN": Checkpoint(gan="ACGAN",
                             root=""),
         "ADCGAN": Checkpoint(gan="ADCGAN",
                              root=""),
         "BigGAN": Checkpoint(gan="BigGAN",
                              root=""),
         "ContraGAN": Checkpoint(gan="ContraGAN",
                                 root=""),
         "MHGAN": Checkpoint(gan="MHGAN",
                             root=""),
         "ReACGAN": Checkpoint(gan="ReACGAN",
                               root=""),
         "cStyleGAN2": Checkpoint(gan="cStyleGAN2",
                                  root=""),
         }

datasets = []

dataset_save_root = ""


class RngManager:
    def __init__(self, seeds):
        self.seeds = seeds
        self.current_seed = None

    def next_seed(self):
        self.current_seed = self.seeds.pop()
        self.set_seed(self.current_seed)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def get_current_seed(self):
        return self.current_seed


class FakeDatasetGenerator:
    def __init__(self, gan, config_paths, checkpoint_paths):
        self.number = 20
        self.gen = aug.Augment(1, gan, "identity", None, config_paths, checkpoint_paths, device)

    def next_gan(self):
        self.gen.next_gan()

    def get_img(self, label):
        return self.gen.forward(None, label)

    def get_dataset_like(self, dataset, first_save_path):
        try:
            shutil.copytree(os.path.join(dataset, "train"), os.path.join(first_save_path, "train"))
        except:
            pass

        mkdir(os.path.join(first_save_path, "valid"))

        # Generate dataset
        for cls in os.listdir(os.path.join(dataset, "train")):
            mkdir(os.path.join(first_save_path, "valid", cls))

            cls_path = os.path.join(dataset, "train", cls)
            cls_count = len(os.listdir(cls_path)) * self.multiplier

            for i in range(cls_count):
                fake_img = self.get_img(labels_map[cls])
                fake_img.save(os.path.join(first_save_path, "valid", cls, f"{i}.png"))


def mkdir(p):
    try:
        os.mkdir(p)
    except:
        pass


def gen_dataset(gan, real_datasets, config_paths, checkpoint_paths, save_path):
    print(real_datasets, config_paths, checkpoint_paths)

    mkdir(save_path)

    fake_dataset_generator = FakeDatasetGenerator(gan, config_paths, checkpoint_paths)

    # Train real Eval fake
    first_save_path = os.path.join(save_path, "train_real_eval_fake")
    mkdir(first_save_path)

    for i, dataset in enumerate(real_datasets):
        fake_dataset_generator.next_gan()

        _first_save_path = os.path.join(first_save_path, str(i))

        mkdir(_first_save_path)

        fake_dataset_generator.get_dataset_like(dataset, _first_save_path)


for gan, ckpt in saves.items():
    rng = RngManager([42])
    rng.next_seed()

    config_paths = []
    checkpoint_paths = []

    for split_name in os.listdir(ckpt.root):
        split = os.path.join(ckpt.root, split_name)

        # Get config file
        config_file = glob.glob(os.path.join(split, "*.yaml"))[-1]
        config_paths.append(config_file)

        # Get latest checkpoint
        checkpoint_path = os.path.join(split, "checkpoints", sorted(os.listdir(os.path.join(split, "checkpoints")))[-1])
        checkpoint_paths.append(checkpoint_path)

    config_paths = sorted(config_paths)
    checkpoint_paths = sorted(checkpoint_paths)

    gen_dataset(gan, datasets, config_paths, checkpoint_paths, os.path.join(dataset_save_root, gan))
