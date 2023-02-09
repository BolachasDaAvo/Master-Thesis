import glob
import os
from collections import namedtuple

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import augs.ganaug.selectors.identity as identity
from augs.ganaug.generators.studiogan.gen import ImageGenerator

"""
Existe uma diferença nas metricas calculadas pelo Pytorch-StudioGAN e por este codigo, devido aos random flips horizontais
"""


class Metrics:
    def __init__(self, real_images, device, model_type, model_path=None):
        self.batch_size = 32
        self.real_images = real_images
        self.device = device

        # Model
        self.save_output = Metrics.SaveOutput()
        if model_type == "efficientnet":
            self.model = self._get_efficientnet_model(model_path)
            for name, layer in self.model.named_children():
                if name == "classifier":
                    layer.fc.register_forward_pre_hook(self.save_output)
            self.preprocess = transforms.Compose([])
        elif model_type == "inception":
            self.model = self._get_inception_model(model_path)
            for name, layer in self.model.named_children():
                if name == "fc":
                    layer.register_forward_pre_hook(self.save_output)
            if model_path:
                self.preprocess = transforms.Compose([
                    transforms.Resize(299),
                ])
            else:
                self.preprocess = transforms.Compose([
                    transforms.Resize(299),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            raise NotImplementedError

        _, self.real_features = self._get_logits_features(self.real_images)

    def _get_efficientnet_model(self, model_path):
        model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b4",
                               pretrained=False if model_path else True)
        if model_path:
            model.classifier.fc = nn.Linear(model.classifier.fc.in_features, 4)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        model = model.to(self.device).eval()
        return model

    def _get_inception_model(self, model_path):
        model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=False if model_path else True)
        if model_path:
            model.fc = nn.Linear(model.fc.in_features, 4)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        model = model.to(self.device).eval()
        return model

    def _get_logits_features(self, images):
        images = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
        features_list = []
        logits_list = []
        for image in images:
            logits, features = self._get_outputs(image)
            logits_list.append(logits)
            features_list.append(features)

        return torch.cat(logits_list, dim=0), torch.cat(features_list, dim=0)

    def _get_outputs(self, images):
        input_batch = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)

        logits = self.model(input_batch)
        features = self.save_output.outputs[0][0].to(self.device)
        self.save_output.clear()
        return logits.detach().cpu(), features.detach().cpu()

    def _compute_pairwise_distance(self, data_x, data_y=None):
        if data_y is None:
            data_y = data_x
        dists = metrics.pairwise_distances(data_x, data_y, metric="euclidean")
        return dists

    def _get_kth_value(self, unsorted, k, axis=-1):
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallest = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallest.max(axis=axis)
        return kth_values

    def _compute_nearest_neighbour_distances(self, input_features, nearest_k):
        distances = self._compute_pairwise_distance(input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def calc_prdc(self, images, nearest_k=3):
        _, fake_features = self._get_logits_features(images)
        fake_features = fake_features.numpy()
        real_features = self.real_features.numpy()

        real_nearest_neighbour_distances = self._compute_nearest_neighbour_distances(real_features, nearest_k)
        fake_nearest_neighbour_distances = self._compute_nearest_neighbour_distances(fake_features, nearest_k)
        distance_real_fake = self._compute_pairwise_distance(real_features, fake_features)

        precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()
        recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()
        density = (1. / float(nearest_k)) * (
                distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0).mean()
        coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

        return precision, recall, density, coverage

    def _cal_kl_div(self, probs, splits):
        scores = []
        num_samples = probs.shape[0]
        with torch.no_grad():
            for j in range(splits):
                part = probs[(j * num_samples // splits):((j + 1) * num_samples // splits), :]
                kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
                kl = torch.mean(torch.sum(kl, 1))
                kl = torch.exp(kl)
                scores.append(kl.unsqueeze(0))

            scores = torch.cat(scores, 0)
            m_scores = torch.mean(scores).detach().cpu().numpy()
            m_std = torch.std(scores).detach().cpu().numpy()
        return m_scores, m_std

    def calc_is(self, images, splits=1, eps=1e-6):
        fake_logits, _ = self._get_logits_features(images)
        fake_probs = torch.nn.functional.softmax(fake_logits, dim=1)
        fake_probs[fake_probs < eps] = eps
        inception_score, _ = self._cal_kl_div(fake_probs, splits=splits)

        return inception_score

    def calc_fid(self, images, eps=1e-6):
        _, fake_features = self._get_logits_features(images)
        fake_features = fake_features.numpy().astype(np.float64)
        real_features = self.real_features.numpy().astype(np.float64)

        mu1 = np.mean(real_features, axis=0)
        mu1 = np.atleast_1d(mu1)

        sigma1 = np.cov(real_features, rowvar=False)
        sigma1 = np.atleast_2d(sigma1)

        mu2 = np.mean(fake_features, axis=0)
        mu2 = np.atleast_1d(mu2)

        sigma2 = np.cov(fake_features, rowvar=False)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

        return fid

    class SaveOutput:
        def __init__(self):
            self.outputs = []

        def __call__(self, module, module_input):
            self.outputs.append(module_input)

        def clear(self):
            self.outputs = []


def get_labels(size, num_labels):
    size_per_label = size // num_labels
    labels = []
    for label in range(num_labels):
        labels.extend([label] * size_per_label)

    return labels


def get_metrics():
    for gan, ckpt in saves.items():
        print(f"Starting {gan}...")

        config_paths = []
        checkpoint_paths = []

        for split_name in os.listdir(ckpt.root):
            split = os.path.join(ckpt.root, split_name)

            # Get config file
            config_file = glob.glob(os.path.join(split, "*.yaml"))[-1]
            config_paths.append(config_file)

            # Get latest checkpoint
            """
            checkpoint_path = os.path.join(split, "checkpoints",
                                           sorted(os.listdir(os.path.join(split, "checkpoints")))[-1])
            checkpoint_paths.append(checkpoint_path)
            """
            checkpoint_paths.append(split)

        config_paths = sorted(config_paths)
        checkpoint_paths = sorted(checkpoint_paths)

        fid_inception_imagenet_list = []
        is_inception_imagenet_list = []
        recall_inception_imagenet_list = []
        precision_inception_imagenet_list = []
        density_inception_imagenet_list = []
        coverage_inception_imagenet_list = []

        fid_inception_dataset_list = []
        is_inception_dataset_list = []
        recall_inception_dataset_list = []
        precision_inception_dataset_list = []
        density_inception_dataset_list = []
        coverage_inception_dataset_list = []

        intrafid_inception_imagenet_list = [[], [], [], []]
        intrafid_inception_dataset_list = [[], [], [], []]

        for cfg_path, ckpt_path, model_path, images, labels in zip(config_paths, checkpoint_paths, model_paths,
                                                                   real_images, real_labels):
            generator = ImageGenerator(cfg_path, ckpt_path, identity.Selector(), device)
            fake_images = [generator.forward(label) for label in tqdm(get_labels(2500, 4))]

            # IntraFID
            print("Starting IntraFID...")
            _, real_idx = np.unique(np.array(labels), return_index=True)
            _, fake_idx = np.unique(np.array(get_labels(10000, 4)), return_index=True)
            for cls in range(4):
                metrics_inception_imagenet = Metrics(
                    images[real_idx[cls]: real_idx[cls + 1] if cls + 1 < 4 else len(images)], device, "inception", None)
                fid = metrics_inception_imagenet.calc_fid(
                    fake_images[fake_idx[cls]: fake_idx[cls + 1] if cls + 1 < 4 else len(fake_images)])
                intrafid_inception_imagenet_list[cls].append(fid)

                metrics_inception_dataset = Metrics(
                    images[real_idx[cls]: real_idx[cls + 1] if cls + 1 < 4 else len(images)], device, "inception",
                    model_path)
                fid = metrics_inception_dataset.calc_fid(
                    fake_images[fake_idx[cls]: fake_idx[cls + 1] if cls + 1 < 4 else len(fake_images)])
                intrafid_inception_dataset_list[cls].append(fid)

            metrics_inception_imagenet = Metrics(images, device, "inception", None)
            metrics_inception_dataset = Metrics(images, device, "efficientnet", model_path)

            # FID and IS
            print("Starting FID and IS...")
            inception_score = metrics_inception_imagenet.calc_is(fake_images)
            fid = metrics_inception_imagenet.calc_fid(fake_images)
            fid_inception_imagenet_list.append(fid)
            is_inception_imagenet_list.append(inception_score)
            inception_score = metrics_inception_dataset.calc_is(fake_images)
            fid = metrics_inception_dataset.calc_fid(fake_images)
            fid_inception_dataset_list.append(fid)
            is_inception_dataset_list.append(inception_score)

            # Improved Precision and Recall
            print("Starting Improved Precision and Recall...")
            generator = ImageGenerator(cfg_path, ckpt_path, identity.Selector(), device)
            for _ in tqdm(range(10)):
                fake_images = [generator.forward(label) for label in get_labels(len(labels), 4)]

                # InceptionV3 trained on ImageNet
                precision, recall, density, coverage = metrics_inception_imagenet.calc_prdc(fake_images)
                recall_inception_imagenet_list.append(recall)
                precision_inception_imagenet_list.append(precision)
                density_inception_imagenet_list.append(density)
                coverage_inception_imagenet_list.append(coverage)

                # InceptionV3 trained on Dataset
                precision, recall, density, coverage = metrics_inception_dataset.calc_prdc(fake_images)
                recall_inception_dataset_list.append(recall)
                precision_inception_dataset_list.append(precision)
                density_inception_dataset_list.append(density)
                coverage_inception_dataset_list.append(coverage)

        print(
            f"InceptionV3 trained on Imagenet: FID & IS & Precision & Recall: {np.mean(fid_inception_imagenet_list):.2f} & {np.mean(is_inception_imagenet_list):.2f} & {np.mean(precision_inception_imagenet_list):.2f} & {np.mean(recall_inception_imagenet_list):.2f}")
        print(f"IntraFID: {np.mean(np.array(intrafid_inception_imagenet_list), axis=1)}")
        print(
            f"InceptionV3 trained on Dataset: FID & IS & Precision & Recall: {np.mean(fid_inception_dataset_list):.2f} & {np.mean(is_inception_dataset_list):.2f} & {np.mean(precision_inception_dataset_list):.2f} & {np.mean(recall_inception_dataset_list):.2f}")
        print(f"IntraFID: {np.mean(np.array(intrafid_inception_dataset_list), axis=1)}")


Checkpoint = namedtuple("Checkpoint", ["gan", "root"])

saves = {"ADCGAN": Checkpoint(gan="ADCGAN",
                              root=""),
         "ACGAN": Checkpoint(gan="ACGAN",
                             root=""),
         "BigGAN": Checkpoint(gan="BigGAN",
                              root=""),
         "ContraGAN": Checkpoint(gan="ContraGAN",
                                 root=""),
         "MHGAN": Checkpoint(gan="MHGAN",
                             root=""),
         "ProjGAN": Checkpoint(gan="ProjGAN",
                               root=""),
         "ReACGAN": Checkpoint(gan="ReACGAN",
                               root=""),
         "cStyleGAN2": Checkpoint(gan="cStyleGAN2",
                                  root=""),
         }

dataset_paths = []

model_paths = []

real_images = []
real_labels = []
for dataset_path in dataset_paths:
    real_images_split = []
    real_labels_split = []
    for img_name, label in ImageFolder(dataset_path).imgs:
        real_images_split.append(transforms.ToTensor()(Image.open(img_name).convert("RGB")))
        real_labels_split.append(label)

    real_images.append(real_images_split)
    real_labels.append(real_labels_split)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
get_metrics()
