import numpy as np
import sklearn.metrics as metrics
import torch
from torchvision import transforms


class Selector:
    def __init__(self, real_images, device, *, k=5):
        self.batch_size = 4
        self.real_images = real_images
        self.device = device

        # Model
        self.save_output = SaveOutput()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True).to(self.device)
        self.model.eval()
        for name, layer in self.model.named_children():
            if name == "fc":
                layer.register_forward_pre_hook(self.save_output)

        # KNN
        self.real_features = self._get_features(self.real_images).numpy()
        self.real_nearest_neighbour_distances = self._compute_nearest_neighbour_distances(self.real_features, k)
        median = self.real_nearest_neighbour_distances < np.median(self.real_nearest_neighbour_distances)
        self.real_nearest_neighbour_distances = self.real_nearest_neighbour_distances[median]
        self.real_features = self.real_features[median]

    def _get_features(self, images):
        images = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
        features = torch.Tensor()
        for image in images:
            features = torch.cat((features, self._get_outputs(image)), dim=0)

        return features

    def _get_outputs(self, images):
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_batch = torch.stack([preprocess(image) for image in images], dim=0).to(self.device)

        self.model(input_batch)
        features = self.save_output.outputs[0][0].to(self.device)
        self.save_output.clear()
        return features.detach().cpu()

    def _compute_pairwise_distance(self, data_x, data_y=None):
        if data_y is None:
            data_y = data_x
        dists = metrics.pairwise_distances(data_x, data_y, metric='euclidean', n_jobs=8)
        return dists

    def _get_kth_value(self, unsorted, k, axis=-1):
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def _compute_nearest_neighbour_distances(self, input_features, nearest_k):
        distances = self._compute_pairwise_distance(input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def select(self, images):
        fake_features = self._get_features(images).numpy()
        distance_real_fake = self._compute_pairwise_distance(fake_features, self.real_features)

        realism = []
        for distance in distance_real_fake:
            distance[distance <= 0] = 1e-05
            realism.append(np.max(self.real_nearest_neighbour_distances / distance))

        return [image for i, image in enumerate(images) if realism[i] >= 1]


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []
