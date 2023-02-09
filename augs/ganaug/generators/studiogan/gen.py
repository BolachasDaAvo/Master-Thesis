"""
This code uses code borrowed from https://github.com/POSTECH-CVLab/PyTorch-StudioGAN with some modifications

The MIT License (MIT)

PyTorch StudioGAN:
Copyright (c) 2020 MinGuk Kang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.nn.functional import one_hot

import augs.ganaug.generators.studiogan.utils.misc as misc
import augs.ganaug.generators.studiogan.utils.ops as ops


class ImageGenerator(torch.nn.Module):
    def __init__(self, config_path, checkpoint_path, selector, device):
        super().__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.selector = selector
        self.device = device
        self.config = misc.EmptyObject()
        self.gen = None
        self.gen_mapping = None
        self.gen_synthesis = None

        self._load_config()
        self._load_generator()
        self._load_checkpoint()

        self.imgs = {i: [] for i in range(self.config.DATA.num_classes)}
        self.img_batch = 256
        self.batch_size = 8

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
            # Range [0,1] -> PIL
            # img = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            # pil_imgs.append(Image.fromarray(img))
            # Range [-1,1] -> PIL
            # img = img.add(1).div(2).mul(255.0).add(0.5).clamp(0.0, 255.0).byte().permute(1, 2, 0).cpu().numpy()
            # pil_imgs.append(Image.fromarray(img))
            # Range [-1,1] -> [1,0]
            img = img.add(1).div(2).clamp(0.0, 1.0).cpu()
            pil_imgs.append(img)
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

    def forward(self, label):
        while len(self.imgs[label]) <= 0:
            self._get_img_batch(label)
            self.imgs[label] = self.selector.select(self.imgs[label])

        img = self.imgs[label].pop()
        return img
