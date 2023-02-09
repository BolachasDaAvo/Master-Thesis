import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import pairwise_distances
from torchvision import transforms
from torchvision.datasets import ImageFolder


class FeatureExtractor:
    def __init__(self, device, model_path=None):
        self.batch_size = 4
        self.device = device

        # Model
        self.save_output = FeatureExtractor.SaveOutput()
        self.model = self._get_model(model_path)
        for name, layer in self.model.named_children():
            if name == "classifier":
                layer.fc.register_forward_pre_hook(self.save_output)

    def _get_model(self, model_path):
        model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b4",
                               pretrained=True if model_path else False)
        model.classifier.fc = nn.Linear(model.classifier.fc.in_features, 4)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        model = model.to(self.device).eval()
        return model

    def _get_outputs(self, images):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        input_batch = torch.stack([preprocess(image) for image in images], dim=0).to(self.device)

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


def calc_distance(features_x, features_y):
    distances = pairwise_distances(features_x, features_y, metric="euclidean")

    return distances


def delete_diagonal(array):
    return array[~np.eye(array.shape[0], dtype=bool)].reshape(array.shape[0], -1)


def average_distance(distances):
    return np.mean(distances.flatten())


def average_min_distance(distances):
    return np.mean(np.min(distances, axis=1).flatten())


def get_dataset_features(dataset_path, feature_extractor):
    images = []
    for img_name, label in ImageFolder(dataset_path).imgs:
        images.append(Image.open(img_name).convert("RGB"))

    features = feature_extractor.load_fake_features(images)

    return features


def get_dataset_cls_features(dataset_path, cls, feature_extractor):
    images = []
    for img_name, label in ImageFolder(dataset_path).imgs:
        if label == cls:
            images.append(Image.open(img_name).convert("RGB"))

    features = feature_extractor.load_fake_features(images)

    return features


def get_fake_features(labels, image_generator, feature_extractor):
    images = [image_generator.forward(label) for label in labels]

    features = feature_extractor.load_fake_features(images)

    return features


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_paths = [""]

model_paths = [""]

real_images = {0: [], 1: [], 2: [], 3: []}
for img_name, label in ImageFolder("").imgs:
    real_images[label].append(Image.open(img_name).convert("RGB"))


def feature_distances():
    pass