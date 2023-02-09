import argparse
import glob
import os
import os.path
import pickle
import random
from collections import namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torchvision.transforms as tfs
import wandb
import yaml
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.functional import one_hot
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import augs.ganaug.generators.studiogan.utils.misc as misc
import augs.ganaug.generators.studiogan.utils.ops as ops


class Augmenter(torch.nn.Module):
    def __init__(self, p, gan, gan_config_paths, gan_checkpoint_paths, classifier_checkpoint_paths, features_path,
                 labels_path, device):
        super().__init__()
        self.p = p
        self.gan = gan
        self.gan_config_paths = gan_config_paths
        self.gan_checkpoint_paths = gan_checkpoint_paths
        self.classifier_checkpoint_paths = classifier_checkpoint_paths
        self.features_paths = features_path
        self.labels_paths = labels_path
        self.device = device
        self.current = -1
        self.gen = None

    def next_generator(self):
        self.current = self.current + 1 if self.current + 1 < len(self.gan_config_paths) else 0

        self.gen = FilteredImageGenerator(self.gan_config_paths[self.current], self.gan_checkpoint_paths[self.current],
                                          self.features_paths[self.current], self.labels_paths[self.current],
                                          self.classifier_checkpoint_paths[self.current], self.device)

        return self.gen

    def forward(self, img, label):
        if torch.rand((1,)) > self.p:
            return img

        return self.gen.forward(label)


class FilteredImageGenerator(torch.nn.Module):
    def __init__(self, config_path, checkpoint_path, features_path, labels_path, classifier_path, device):
        super().__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.features_path = features_path
        self.labels_path = labels_path
        self.classifier_path = classifier_path
        self.feature_extractor = FeatureExtractor(device, classifier_path)
        self.features = None
        self.labels = None
        self.device = device
        self.config = misc.EmptyObject()
        self.gen = None
        self.gen_mapping = None
        self.gen_synthesis = None

        self._load_features_labels()
        self._load_config()
        self._load_generator()
        self._load_checkpoint()

        self.imgs = {i: [] for i in range(self.config.DATA.num_classes)}
        self.img_batch = 256
        self.batch_size = 8
        self.ratio = 0.7
        self.k = 30

    def _load_features_labels(self):
        with open(self.features_path, "rb") as f:
            self.features = pickle.load(f)

        with open(self.labels_path, "rb") as f:
            self.labels = pickle.load(f)

    def _load_config(self):
        self.config.DATA = misc.EmptyObject()

        self.config.DATA.name = "CIFAR10"
        self.config.DATA.img_size = 32
        self.config.DATA.num_classes = 10
        self.config.DATA.img_channels = 3

        self.config.MODEL = misc.EmptyObject()

        self.config.MODEL.backbone = "resnet"
        self.config.MODEL.g_cond_mtd = "W/O"
        self.config.MODEL.d_cond_mtd = "W/O"
        self.config.MODEL.aux_cls_type = "W/O"
        self.config.MODEL.normalize_d_embed = False
        self.config.MODEL.d_embed_dim = "N/A"
        self.config.MODEL.apply_g_sn = False
        self.config.MODEL.apply_d_sn = False
        self.config.MODEL.g_act_fn = "ReLU"
        self.config.MODEL.d_act_fn = "ReLU"
        self.config.MODEL.apply_attn = False
        self.config.MODEL.attn_g_loc = ["N/A"]
        self.config.MODEL.attn_d_loc = ["N/A"]
        self.config.MODEL.z_prior = "gaussian"
        self.config.MODEL.z_dim = 128
        self.config.MODEL.w_dim = "N/A"
        self.config.MODEL.g_shared_dim = "N/A"
        self.config.MODEL.g_conv_dim = 64
        self.config.MODEL.d_conv_dim = 64
        self.config.MODEL.g_depth = "N/A"
        self.config.MODEL.d_depth = "N/A"
        self.config.MODEL.apply_g_ema = False
        self.config.MODEL.g_ema_decay = "N/A"
        self.config.MODEL.g_ema_start = "N/A"
        self.config.MODEL.g_init = "ortho"
        self.config.MODEL.d_init = "ortho"
        self.config.MODEL.info_type = "N/A"
        self.config.MODEL.g_info_injection = "N/A"
        self.config.MODEL.info_num_discrete_c = "N/A"
        self.config.MODEL.info_num_conti_c = "N/A"
        self.config.MODEL.info_dim_discrete_c = "N/A"

        self.config.LOSS = misc.EmptyObject()

        self.config.LOSS.adv_loss = "vanilla"
        self.config.LOSS.cond_lambda = "N/A"
        self.config.LOSS.tac_gen_lambda = "N/A"
        self.config.LOSS.tac_dis_lambda = "N/A"
        self.config.LOSS.mh_lambda = "N/A"
        self.config.LOSS.apply_fm = False
        self.config.LOSS.fm_lambda = "N/A"
        self.config.LOSS.apply_r1_reg = False
        self.config.LOSS.r1_place = "N/A"
        self.config.LOSS.r1_lambda = "N/A"
        self.config.LOSS.m_p = "N/A"
        self.config.LOSS.temperature = "N/A"
        self.config.LOSS.apply_wc = False
        self.config.LOSS.wc_bound = "N/A"
        self.config.LOSS.apply_gp = False
        self.config.LOSS.gp_lambda = "N/A"
        self.config.LOSS.apply_dra = False
        self.config.LOSS.dra_lambda = "N/A"
        self.config.LOSS.apply_maxgp = False
        self.config.LOSS.maxgp_lambda = "N/A"
        self.config.LOSS.apply_cr = False
        self.config.LOSS.cr_lambda = "N/A"
        self.config.LOSS.apply_bcr = False
        self.config.LOSS.real_lambda = "N/A"
        self.config.LOSS.fake_lambda = "N/A"
        self.config.LOSS.apply_zcr = False
        self.config.LOSS.radius = "N/A"
        self.config.LOSS.g_lambda = "N/A"
        self.config.LOSS.d_lambda = "N/A"
        self.config.LOSS.apply_lo = False
        self.config.LOSS.lo_alpha = "N/A"
        self.config.LOSS.lo_beta = "N/A"
        self.config.LOSS.lo_rate = "N/A"
        self.config.LOSS.lo_lambda = "N/A"
        self.config.LOSS.lo_steps4train = "N/A"
        self.config.LOSS.lo_steps4eval = "N/A"
        self.config.LOSS.apply_topk = False
        self.config.LOSS.topk_gamma = "N/A"
        self.config.LOSS.topk_nu = "N/A"
        self.config.LOSS.infoGAN_loss_discrete_lambda = "N/A"
        self.config.LOSS.infoGAN_loss_conti_lambda = "N/A"
        self.config.LOSS.apply_lecam = False
        self.config.LOSS.lecam_lambda = "N/A"
        self.config.LOSS.lecam_ema_start_iter = "N/A"
        self.config.LOSS.lecam_ema_decay = "N/A"
        self.config.LOSS.t_start_step = "N/A"
        self.config.LOSS.t_end_step = "N/A"

        self.config.OPTIMIZATION = misc.EmptyObject()

        self.config.OPTIMIZATION.type_ = "Adam"
        self.config.OPTIMIZATION.batch_size = 64
        self.config.OPTIMIZATION.acml_steps = 1
        self.config.OPTIMIZATION.g_lr = 0.0002
        self.config.OPTIMIZATION.d_lr = 0.0002
        self.config.OPTIMIZATION.g_weight_decay = 0.0
        self.config.OPTIMIZATION.d_weight_decay = 0.0
        self.config.OPTIMIZATION.momentum = "N/A"
        self.config.OPTIMIZATION.nesterov = "N/A"
        self.config.OPTIMIZATION.alpha = "N/A"
        self.config.OPTIMIZATION.beta1 = 0.5
        self.config.OPTIMIZATION.beta2 = 0.999
        self.config.OPTIMIZATION.d_first = True
        self.config.OPTIMIZATION.g_updates_per_step = 1
        self.config.OPTIMIZATION.d_updates_per_step = 5
        self.config.OPTIMIZATION.total_steps = 100000

        self.config.PRE = misc.EmptyObject()

        self.config.PRE.apply_rflip = True

        self.config.AUG = misc.EmptyObject()

        self.config.AUG.apply_diffaug = False
        self.config.AUG.apply_ada = False
        self.config.AUG.ada_initial_augment_p = "N/A"
        self.config.AUG.ada_target = "N/A"
        self.config.AUG.ada_kimg = "N/A"
        self.config.AUG.ada_interval = "N/A"
        self.config.AUG.apply_apa = False
        self.config.AUG.apa_initial_augment_p = "N/A"
        self.config.AUG.apa_target = "N/A"
        self.config.AUG.apa_kimg = "N/A"
        self.config.AUG.apa_interval = "N/A"
        self.config.AUG.cr_aug_type = "W/O"
        self.config.AUG.bcr_aug_type = "W/O"
        self.config.AUG.diffaug_type = "W/O"
        self.config.AUG.ada_aug_type = "W/O"

        self.config.STYLEGAN = misc.EmptyObject()

        self.config.STYLEGAN.stylegan3_cfg = "N/A"
        self.config.STYLEGAN.cond_type = ["PD", "SPD", "2C", "D2DCE"]
        self.config.STYLEGAN.g_reg_interval = "N/A"
        self.config.STYLEGAN.d_reg_interval = "N/A"
        self.config.STYLEGAN.mapping_network = "N/A"
        self.config.STYLEGAN.style_mixing_p = "N/A"
        self.config.STYLEGAN.g_ema_kimg = "N/A"
        self.config.STYLEGAN.g_ema_rampup = "N/A"
        self.config.STYLEGAN.apply_pl_reg = False
        self.config.STYLEGAN.pl_weight = "N/A"
        self.config.STYLEGAN.d_architecture = "N/A"
        self.config.STYLEGAN.d_epilogue_mbstd_group_size = "N/A"
        self.config.STYLEGAN.blur_init_sigma = "N/A"

        self.config.RUN = misc.EmptyObject()

        self.config.MISC = misc.EmptyObject()

        self.config.MISC.no_proc_data = ["CIFAR10", "CIFAR100", "Tiny_ImageNet"]
        self.config.MISC.base_folders = ["checkpoints", "figures", "logs", "moments", "samples", "values"]
        self.config.MISC.classifier_based_GAN = ["AC", "2C", "D2DCE"]
        self.config.MISC.info_params = ["info_discrete_linear", "info_conti_mu_linear", "info_conti_var_linear"]
        self.config.MISC.cas_setting = {
            "CIFAR10": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 32,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
            "Tiny_ImageNet": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 34,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
            "ImageNet": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 34,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
        }

        self.config.MODULES = misc.EmptyObject()

        super_cfgs = {
            "DATA": self.config.DATA,
            "MODEL": self.config.MODEL,
            "LOSS": self.config.LOSS,
            "OPTIMIZATION": self.config.OPTIMIZATION,
            "PRE": self.config.PRE,
            "AUG": self.config.AUG,
            "RUN": self.config.RUN,
            "STYLEGAN": self.config.STYLEGAN
        }

        with open(self.config_path, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
            for super_cfg_name, attr_value in yaml_cfg.items():
                for attr, value in attr_value.items():
                    if hasattr(super_cfgs[super_cfg_name], attr):
                        setattr(super_cfgs[super_cfg_name], attr, value)
                    else:
                        raise AttributeError(
                            "There does not exist '{cls}.{attr}' attribute in the config.py.".format(cls=super_cfg_name,
                                                                                                     attr=attr))

        if self.config.MODEL.apply_g_sn:
            self.config.MODULES.g_conv2d = ops.snconv2d
            self.config.MODULES.g_deconv2d = ops.sndeconv2d
            self.config.MODULES.g_linear = ops.snlinear
            self.config.MODULES.g_embedding = ops.sn_embedding
        else:
            self.config.MODULES.g_conv2d = ops.conv2d
            self.config.MODULES.g_deconv2d = ops.deconv2d
            self.config.MODULES.g_linear = ops.linear
            self.config.MODULES.g_embedding = ops.embedding

        if self.config.MODEL.apply_d_sn:
            self.config.MODULES.d_conv2d = ops.snconv2d
            self.config.MODULES.d_deconv2d = ops.sndeconv2d
            self.config.MODULES.d_linear = ops.snlinear
            self.config.MODULES.d_embedding = ops.sn_embedding
        else:
            self.config.MODULES.d_conv2d = ops.conv2d
            self.config.MODULES.d_deconv2d = ops.deconv2d
            self.config.MODULES.d_linear = ops.linear
            self.config.MODULES.d_embedding = ops.embedding

        if self.config.MODEL.g_cond_mtd == "cBN" or self.config.MODEL.g_info_injection == "cBN" or self.config.MODEL.backbone == "big_resnet":
            self.config.MODULES.g_bn = ops.ConditionalBatchNorm2d
        elif self.config.MODEL.g_cond_mtd == "W/O":
            self.config.MODULES.g_bn = ops.batchnorm_2d
        elif self.config.MODEL.g_cond_mtd == "cAdaIN":
            pass
        else:
            raise NotImplementedError

        if not self.config.MODEL.apply_d_sn:
            self.config.MODULES.d_bn = ops.batchnorm_2d

        if self.config.MODEL.g_act_fn == "ReLU":
            self.config.MODULES.g_act_fn = nn.ReLU(inplace=True)
        elif self.config.MODEL.g_act_fn == "Leaky_ReLU":
            self.config.MODULES.g_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.config.MODEL.g_act_fn == "ELU":
            self.config.MODULES.g_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.config.MODEL.g_act_fn == "GELU":
            self.config.MODULES.g_act_fn = nn.GELU()
        elif self.config.MODEL.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError

        if self.config.MODEL.d_act_fn == "ReLU":
            self.config.MODULES.d_act_fn = nn.ReLU(inplace=True)
        elif self.config.MODEL.d_act_fn == "Leaky_ReLU":
            self.config.MODULES.d_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.config.MODEL.d_act_fn == "ELU":
            self.config.MODULES.d_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.config.MODEL.d_act_fn == "GELU":
            self.config.MODULES.d_act_fn = nn.GELU()
        elif self.config.MODEL.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError

    def _load_generator(self):
        module = __import__(f"augs.ganaug.generators.studiogan.models.{self.config.MODEL.backbone}",
                            fromlist=["something"])

        if self.config.MODEL.backbone in ["stylegan2", "stylegan2_transitional"]:
            self.gen = module.Generator(z_dim=self.config.MODEL.z_dim,
                                        c_dim=self.config.DATA.num_classes,
                                        w_dim=self.config.MODEL.w_dim,
                                        img_resolution=self.config.DATA.img_size,
                                        img_channels=self.config.DATA.img_channels,
                                        MODEL=self.config.MODEL,
                                        mapping_kwargs={"num_layers": self.config.STYLEGAN.mapping_network},
                                        synthesis_kwargs={"channel_base": 16384, "channel_max": 512,
                                                          "num_fp16_res": 0, "conv_clamp": None}).to(self.device)
            self.gen_mapping, self.gen_synthesis = self.gen.mapping, self.gen.synthesis
        else:
            self.gen = module.Generator(z_dim=self.config.MODEL.z_dim,
                                        g_shared_dim=self.config.MODEL.g_shared_dim,
                                        img_size=self.config.DATA.img_size,
                                        g_conv_dim=self.config.MODEL.g_conv_dim,
                                        apply_attn=self.config.MODEL.apply_attn,
                                        attn_g_loc=self.config.MODEL.attn_g_loc,
                                        g_cond_mtd=self.config.MODEL.g_cond_mtd,
                                        num_classes=self.config.DATA.num_classes,
                                        g_init=self.config.MODEL.g_init,
                                        g_depth=self.config.MODEL.g_depth,
                                        mixed_precision=False,
                                        MODULES=self.config.MODULES,
                                        MODEL=self.config.MODEL).to(self.device)
            self.gen_mapping, self.gen_synthesis = None, None

        self.gen.eval()

    def _load_checkpoint(self):
        gen_ckpt_path = glob.glob(os.path.join(self.checkpoint_path, "model=G_ema-best-weights-step*.pth"))[0]
        ckpt = torch.load(gen_ckpt_path, map_location=self.device)
        self.gen.load_state_dict(ckpt["state_dict"], strict=True)

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

        with torch.no_grad():
            for _ in range(self.img_batch // self.batch_size):

                fake_labels = torch.full([self.batch_size], label, device=self.device)

                if self.config.MODEL.backbone in ["stylegan2", "stylegan2_transitional"]:
                    one_hot_fake_labels = one_hot(fake_labels, num_classes=self.config.DATA.num_classes).to(self.device)
                    ws = self.gen_mapping(torch.randn(self.batch_size, self.config.MODEL.z_dim, device=self.device),
                                          one_hot_fake_labels, truncation_psi=-1, update_emas=False)

                    fake_imgs = self.gen_synthesis(ws, update_emas=False)
                else:
                    fake_imgs = self.gen(torch.randn(self.batch_size, self.config.MODEL.z_dim, device=self.device),
                                         fake_labels, eval=True)

                fake_imgs = self._convert_imgs(fake_imgs)
                self.imgs[label].extend(fake_imgs)

    def _filter_imgs(self, label):
        filtered = []
        unfiltered_features = self.feature_extractor.get_features(self.imgs[label])

        distances = metrics.pairwise_distances(unfiltered_features, self.features, metric="euclidean")
        nearest_neighbours = np.argpartition(distances, self.k + 1, axis=-1)[..., :self.k]

        labels_tile = np.tile(self.labels, (len(unfiltered_features), 1))
        nearest_labels = np.take_along_axis(labels_tile, nearest_neighbours, axis=-1)

        for i, point in enumerate(nearest_labels):
            if np.bincount(point, minlength=4)[label] >= self.ratio * self.k:
                filtered.append(i)

        self.imgs[label] = [self.imgs[label][i] for i in filtered]

    def forward(self, label):
        while len(self.imgs[label]) <= 0:
            self._get_img_batch(label)
            self._filter_imgs(label)

        img = self.imgs[label].pop()
        return img


class FeatureExtractor:
    def __init__(self, device, model_path=None):
        self.batch_size = 64
        self.device = device

        # Model
        self.save_output = FeatureExtractor.SaveOutput()
        self.model = self._get_model(model_path)
        for name, layer in self.model.named_children():
            if name == "classifier":
                layer.fc.register_forward_pre_hook(self.save_output)

    def _get_model(self, model_path):
        model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b4",
                               pretrained=False if model_path else True)
        model.classifier.fc = nn.Linear(model.classifier.fc.in_features, 4)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        model = model.to(self.device).eval()
        return model

    def _get_outputs(self, images):
        preprocess = tfs.Compose([
            tfs.ToTensor(),
        ])

        input_batch = torch.stack([preprocess(image) for image in images], dim=0).to(self.device)

        with torch.no_grad():
            self.model(input_batch)
        features = self.save_output.outputs[0][0].to(self.device)
        self.save_output.clear()
        return features.detach().cpu()

    def get_features(self, images):
        images = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
        features = torch.Tensor()
        for image in images:
            features = torch.cat((features, self._get_outputs(image)), dim=0)

        return features.numpy()

    class SaveOutput:
        def __init__(self):
            self.outputs = []

        def __call__(self, module, module_input):
            self.outputs.append(module_input)

        def clear(self):
            self.outputs = []


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


class InMemoryDataset(torch_data.Dataset):
    def __init__(self, data_path, *, augmenter=None):
        self.data = []
        self.labels = []
        self.transform_list = []

        # Load data in memory
        for img_name, label in ImageFolder(data_path).imgs:
            self.data.append(Image.open(img_name).convert(mode="RGB"))
            self.labels.append(label)

        # Build transforms
        if augmenter:
            self.transform_list.append(augmenter)
        self.transform_list.append(tfs.ToTensor())

    def __getitem__(self, index):
        img = self.data[index]
        label = int(self.labels[index])

        for t in self.transform_list:
            img = t(img, label) if isinstance(t, Augmenter) else t(img)

        return img, label

    def __len__(self):
        return len(self.data)


class MultipleDatasetManager:
    def __init__(self, data_paths, batch_size, num_workers, augmenter=None):
        super(MultipleDatasetManager, self).__init__()
        self.data_paths = data_paths
        self.current_index = -1
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmenter = augmenter
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

    def next_dataset(self):
        self.current_index = self.current_index + 1 if self.current_index + 1 < len(self.data_paths) else 0

        self.train_dataset = InMemoryDataset(os.path.join(self.data_paths[self.current_index], "train"),
                                             augmenter=self.augmenter)
        self.valid_dataset = InMemoryDataset(os.path.join(self.data_paths[self.current_index], "valid"))

        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, shuffle=True, pin_memory=True)
        self.valid_loader = torch_data.DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, shuffle=True, pin_memory=True)

        return self.train_dataset, self.train_loader, self.valid_dataset, self.valid_loader


def build_model():
    model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b4", pretrained=True)
    model.classifier.fc = nn.Linear(model.classifier.fc.in_features, 4)
    model.to(device)
    return model


def log(message):
    log_file.write(str(message) + "\n")
    log_file.flush()


def get_metrics(y_true, y_pred, prefix):
    report_metrics = {}

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=list(range(4)))
    report = metrics.classification_report(y_true, y_pred, labels=list(range(4)), target_names=labels_readable,
                                           output_dict=True)

    names = ["precision", "recall", "f1-score"]
    for label in labels_readable:
        for n in names:
            report_metrics[f"{prefix}_{label}_{n}"] = report[label][n]
    report_metrics[f"{prefix}_accuracy"] = report["accuracy"]

    return confusion_matrix, report_metrics


def draw_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(10, 7))
    sn.heatmap(pd.DataFrame(confusion_matrix, index=labels_readable, columns=labels_readable), annot=True,
               cmap="Blues", fmt="g")
    return fig


def train():
    metrics_df = None
    ModelCheckpoint = namedtuple("ModelCheckpoint", ["state_dict", "accuracy"])
    for run in range(num_runs):

        log(f"Starting run {run}/{num_runs}")
        rng.next_seed()

        for split_num in range(num_splits):

            # Init
            log(f"Starting split {split_num}/{num_splits}")
            if log_wandb:
                wandb.init(project="Classifiers", entity="mike-wazowski", group=group_name, name=f"split-{split_num}")

            augmenter.next_generator()

            train_set, train_loader, val_set, val_loader = data.next_dataset()
            model = build_model()
            best_model = ModelCheckpoint({}, 0.0)
            loss_func = nn.CrossEntropyLoss()
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, weight_decay=5e-6)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)
            scaler = GradScaler()

            for epoch in tqdm(range(num_epochs)):

                epoch_metrics = {"epoch": epoch}
                wandb_epoch_metrics = {}

                # train
                model.train()
                running_loss = 0.0
                running_y_true = []
                running_y_pred = []

                optimizer.zero_grad()
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    with autocast():
                        outputs = nn.functional.softmax(model(inputs), dim=1)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_func(outputs, labels)
                        loss = loss / acml_steps

                    scaler.scale(loss).backward()
                    if (i + 1) % acml_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    # save statistics
                    running_loss += loss * inputs.size(0)
                    running_y_true += labels.cpu().tolist()
                    running_y_pred += preds.cpu().tolist()

                scheduler.step()

                # calc statistics
                train_loss = running_loss / len(train_set)
                confusion_matrix, train_metrics = get_metrics(running_y_true, running_y_pred, prefix="train")

                # log values
                epoch_metrics.update(train_metrics)
                if log_wandb:
                    wandb_epoch_metrics["train_loss"] = train_loss
                    wandb_epoch_metrics["train_confusion_matrix"] = wandb.Image(draw_confusion_matrix(confusion_matrix))
                    plt.clf()
                log(f"Run {run} Split {split_num} Epoch {epoch} Train")
                log(confusion_matrix)

                # eval
                model.eval()
                running_y_true = []
                running_y_pred = []

                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        outputs = nn.functional.softmax(model(inputs), dim=1)
                        _, preds = torch.max(outputs, 1)

                    # save statistics
                    running_y_true += labels.cpu().tolist()
                    running_y_pred += preds.cpu().tolist()

                # calc statistics
                confusion_matrix, val_metrics = get_metrics(running_y_true, running_y_pred, prefix="val")

                # log values
                epoch_metrics.update(val_metrics)
                if log_wandb:
                    wandb_epoch_metrics["val_confusion_matrix"] = wandb.Image(draw_confusion_matrix(confusion_matrix))
                    plt.clf()
                log(f"Run {run} Split {split_num} Epoch {epoch} Val")
                log(confusion_matrix)

                # log to wandb
                if log_wandb:
                    wandb_epoch_metrics.update(epoch_metrics)
                    wandb.log(wandb_epoch_metrics, step=epoch)

                # log to df
                if metrics_df is None:
                    metrics_df = pd.DataFrame(columns=list(epoch_metrics.keys()))
                metrics_df = pd.concat([metrics_df, pd.DataFrame({k: [v] for k, v in epoch_metrics.items()})],
                                       ignore_index=True)

                # Save model
                if classifier_save_path and best_model.accuracy < val_metrics["val_accuracy"]:
                    best_model = ModelCheckpoint(deepcopy(model.state_dict()), val_metrics["val_accuracy"])

            if log_wandb:
                wandb.finish(0)

            if classifier_save_path:
                torch.save(model.state_dict(), os.path.join(classifier_save_path, f"ckpt_last_{run}_{split_num}.pt"))
                torch.save(best_model.state_dict, os.path.join(classifier_save_path, f"ckpt_best_{run}_{split_num}.pt"))

    log_file.close()
    metrics_df.to_csv(metrics_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_paths", nargs="+")
    parser.add_argument("-log_path", type=str)
    parser.add_argument("-metrics_path", type=str)
    parser.add_argument("--classifier_save_path", type=str)
    parser.add_argument("-group_name", type=str)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-num_workers", type=int)
    parser.add_argument("-num_epochs", type=int)
    parser.add_argument("-num_splits", type=int)
    parser.add_argument("-num_runs", type=int)
    parser.add_argument("-acml_steps", type=int)
    parser.add_argument("-log_wandb", action="store_true")
    parser.add_argument("-gan", type=str)
    parser.add_argument("-p", type=float)
    parser.add_argument("-gan_checkpoints", nargs="+")
    parser.add_argument("-gan_configs", nargs="+")
    parser.add_argument("-classifier_checkpoints", nargs="+")
    parser.add_argument("-defining_features", nargs="+")
    parser.add_argument("-defining_labels", nargs="+")

    args = parser.parse_args()

    data_paths = args.data_paths
    log_path = args.log_path
    metrics_path = args.metrics_path
    classifier_save_path = args.classifier_save_path
    group_name = args.group_name
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    num_splits = args.num_splits
    num_runs = args.num_runs
    acml_steps = args.acml_steps
    log_wandb = args.log_wandb
    gan = args.gan
    p = args.p
    gan_checkpoints = args.gan_checkpoints
    gan_configs = args.gan_configs
    classifier_checkpoints = args.classifier_checkpoints
    defining_features = args.defining_features
    defining_labels = args.defining_labels

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_file = open(log_path, "w")
    labels_readable = ["covid", "lung_opacity", "normal", "viral_pneumonia"]

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    augmenter = Augmenter(p, gan, gan_configs, gan_checkpoints, classifier_checkpoints, defining_features,
                          defining_labels, device)
    data = MultipleDatasetManager(data_paths, batch_size, num_workers, augmenter)
    rng = RngManager([i for i in range(42, 42 + num_runs)])

    train()
