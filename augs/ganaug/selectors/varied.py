import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torchvision import transforms


class Selector:
    def __init__(self, real_images, device, model_path=None):
        self.batch_size = 4
        self.real_images = real_images
        self.device = device

        # Model
        self.save_output = SaveOutput()
        self.model = self._get_model(model_path)
        for name, layer in self.model.named_children():
            if name == "classifier":
                layer.fc.register_forward_pre_hook(self.save_output)

        self.real_features = self._get_features(self.real_images).numpy()

    def _get_model(self, model_path):
        model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b4",
                               pretrained=False if model_path else True)
        model.classifier.fc = nn.Linear(model.classifier.fc.in_features, 4)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        model = model.to(self.device).eval()
        return model

    def _get_features(self, images):
        images = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
        features = torch.Tensor()
        for image in images:
            features = torch.cat((features, self._get_outputs(image)), dim=0)

        return features

    def _get_outputs(self, images):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        input_batch = torch.stack([preprocess(image) for image in images], dim=0).to(self.device)

        self.model(input_batch)
        features = self.save_output.outputs[0][0].to(self.device)
        self.save_output.clear()
        return features.detach().cpu()

    def _compute_pairwise_distance(self, data_x, data_y=None):
        if data_y is None:
            data_y = data_x
        dists = metrics.pairwise_distances(data_x, data_y, metric="euclidean", n_jobs=8)
        return dists

    def select(self, images, threshold=10):
        fake_features = self._get_features(images).numpy()
        distance_real_fake = self._compute_pairwise_distance(fake_features, self.real_features)
        min_distance_real_fake = np.min(distance_real_fake, axis=1) > threshold

        return [image for i, image in enumerate(images) if min_distance_real_fake[i]]


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []
