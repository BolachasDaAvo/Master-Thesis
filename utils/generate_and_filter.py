import pickle

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

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

        with torch.no_grad():
            self.model(input_batch)
        features = self.save_output.outputs[0][0].to(self.device)
        self.save_output.clear()
        return features.detach().cpu()

    def get_features(self, images):
        images = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]
        features = torch.Tensor()
        for image in tqdm(images):
            features = torch.cat((features, self._get_outputs(image)), dim=0)

        return features.numpy()

    class SaveOutput:
        def __init__(self):
            self.outputs = []

        def __call__(self, module, module_input):
            self.outputs.append(module_input)

        def clear(self):
            self.outputs = []


def store_pickle(file_name, obj):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        obj = pickle.load(f)
    return obj


def filter_images(fake_labels, fake_features):
    ratio = 0.7
    k = 30

    new_indexes = []

    distances = metrics.pairwise_distances(fake_features, fake_features, metric="euclidean")
    nearest_neighbours = np.argpartition(distances, k + 1, axis=-1)[..., :k]

    fake_labels_tile = np.tile(fake_labels, (len(fake_labels), 1))
    nearest_labels = np.take_along_axis(fake_labels_tile, nearest_neighbours, axis=-1)

    for i, point in enumerate(nearest_labels):
        if np.bincount(point, minlength=4)[fake_labels[i]] >= ratio * k:
            new_indexes.append(i)

    print(f"Reduced the number of images to {len(new_indexes)}")

    return np.array(new_indexes)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GAN
gan_save_path = ""
gan_checkpoint_path = ""
gan_config_file = ""

# Classifier
classifier_path = ""

# Save
save_path = ""

# Fake images
fake_images = []
fake_labels = []
for img_name, label in ImageFolder(f"").imgs:
    fake_images.append(Image.open(img_name).convert("RGB"))
    fake_labels.append(label)

feature_extractor = FeatureExtractor(device, classifier_path)
fake_features = feature_extractor.get_features(fake_images)

# Filter and save features
for _ in range(10):
    new_indexes = filter_images(fake_labels, fake_features)
    fake_labels = np.take(fake_labels, new_indexes, axis=0)
    fake_features = np.take(fake_features, new_indexes, axis=0)

store_pickle(f"{save_path}/features.pkl", fake_features)
store_pickle(f"{save_path}/labels.pkl", fake_labels)

"""
# GAN
gan_save_path = "/mnt/cirrus/users/1/2/ist189512/run/saves/BigGAN/"
gan_config_files = []
gan_checkpoint_paths = []
for split_name in os.listdir(gan_save_path):
    split = os.path.join(gan_save_path, split_name)

    # Get config file
    config_file = glob.glob(os.path.join(split, "*.yaml"))[0]
    gan_config_files.append(config_file)

    # Get latest checkpoint
    checkpoint_path = os.path.join(split, "checkpoints", sorted(os.listdir(os.path.join(split, "checkpoints")))[-1])
    gan_checkpoint_paths.append(checkpoint_path)

gan_config_files = sorted(gan_config_files)
gan_checkpoint_paths = sorted(gan_checkpoint_paths)

# Classifier
classifier_paths = ["/mnt/cirrus/users/1/2/ist189512/run/saves/classifiers/baseline/ckpt_best_0_0.pt",
                    "/mnt/cirrus/users/1/2/ist189512/run/saves/classifiers/baseline/ckpt_best_0_1.pt",
                    "/mnt/cirrus/users/1/2/ist189512/run/saves/classifiers/baseline/ckpt_best_0_2.pt",
                    "/mnt/cirrus/users/1/2/ist189512/run/saves/classifiers/baseline/ckpt_best_0_3.pt",
                    "/mnt/cirrus/users/1/2/ist189512/run/saves/classifiers/baseline/ckpt_best_0_4.pt"]

# Saves
save_paths = ["/mnt/cirrus/users/1/2/ist189512/run/data/features/0/",
              "/mnt/cirrus/users/1/2/ist189512/run/data/features/1/",
              "/mnt/cirrus/users/1/2/ist189512/run/data/features/2/",
              "/mnt/cirrus/users/1/2/ist189512/run/data/features/3/",
              "/mnt/cirrus/users/1/2/ist189512/run/data/features/4/"]

# Fake images
for gan_config, gan_checkpoint, classifier, save in zip(gan_config_files, gan_checkpoint_paths, classifier_paths,
                                                        save_paths):
    number_fake_images = 10000
    fake_labels = np.array([0] * 2500 + [1] * 2500 + [2] * 2500 + [3] * 2500)

    # Generate fake images
    generator = ImageGenerator(gan_config, gan_checkpoint, identity.Selector(), device)
    fake_images = [generator.forward(label) for label in tqdm(fake_labels)]

    feature_extractor = FeatureExtractor(device, classifier)
    fake_features = feature_extractor.get_features(fake_images)

    # Filter and save features
    for _ in range(10):
        new_indexes = filter_images(fake_labels, fake_features)
        fake_labels = np.take(fake_labels, new_indexes, axis=0)
        fake_features = np.take(fake_features, new_indexes, axis=0)

    store_pickle(f"{save}/features.pkl", fake_features)
    store_pickle(f"{save}/labels.pkl", fake_labels)
"""
