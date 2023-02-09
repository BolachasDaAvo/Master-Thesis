import cv2
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from augs.ganaug.generators.fastgan.gen import ImageGenerator


class KNN:
    def __init__(self, device, model_type, model_path=None):
        self.batch_size = 32
        self.device = device

        # Model
        self.save_output = KNN.SaveOutput()
        if model_type == "efficientnet":
            self.model = self._get_efficientnet_model(model_path)
            for name, layer in self.model.named_children():
                if name == "classifier":
                    layer.fc.register_forward_pre_hook(self.save_output)
            self.preprocess = transforms.Compose([transforms.ToTensor()])
        elif model_type == "inception":
            self.model = self._get_inception_model(model_path)
            for name, layer in self.model.named_children():
                if name == "fc":
                    layer.register_forward_pre_hook(self.save_output)
            if model_path:
                self.preprocess = transforms.Compose([
                    transforms.Resize(299),
                    transforms.ToTensor(),
                ])
            else:
                self.preprocess = transforms.Compose([
                    transforms.Resize(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            raise NotImplementedError

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

    def get_logits_features(self, images):
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

    def _get_k_smallest(self, array, k):
        distances = [(i, array[i]) for i in range(len(array))]

        distances = sorted(distances, key=lambda x: x[1])
        return distances[:k]

    def knn(self, real_image, fake_images, *, k=10, ground_features=None, fake_features=None):
        if ground_features is None:
            _, ground_features = self.get_logits_features(real_image)
        if fake_features is None:
            _, fake_features = self.get_logits_features(fake_images)

        ground_features = ground_features.numpy()
        fake_features = fake_features.numpy()

        distances = self._compute_pairwise_distance(ground_features, fake_features).squeeze()
        nearest_neighbours = self._get_k_smallest(distances, k)

        return [fake_images[i[0]] for i in nearest_neighbours], [i[1] for i in nearest_neighbours]

    class SaveOutput:
        def __init__(self):
            self.outputs = []

        def __call__(self, module, module_input):
            self.outputs.append(module_input)

        def clear(self):
            self.outputs = []


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_paths = []

checkpoint_paths = []
dataset_path = ""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
number_fake_images = 1000
generator = ImageGenerator(config_paths, checkpoint_paths, device)
knn = KNN(device, "efficientnet")

# Real images
real_images = {0: [], 1: [], 2: [], 3: []}
for img_name, label in ImageFolder(dataset_path).imgs:
    real_images[label].append(Image.open(img_name).convert("RGB"))

# Fake images
to_pil = transforms.ToPILImage()
fake_images = {0: [generator.forward(0) for _ in tqdm(range(number_fake_images))],
               1: [generator.forward(1) for _ in tqdm(range(number_fake_images))],
               2: [generator.forward(2) for _ in tqdm(range(number_fake_images))],
               3: [generator.forward(3) for _ in tqdm(range(number_fake_images))]}

for label in range(0, 4):
    fake_image_batch = fake_images[label]

    nearest_fake_images, nearest_distances = knn.knn([real_images[label][0]], fake_image_batch)

    concat_image = cv2.copyMakeBorder(np.array(real_images[label][0]), 0, 0, 0, 30, cv2.BORDER_CONSTANT,
                                      value=[255, 255, 255])
    for img in nearest_fake_images:
        img = np.array(img)
        concat_image = cv2.hconcat([concat_image, img]) if concat_image is not None else img

    cv2.imwrite(f"", concat_image)

