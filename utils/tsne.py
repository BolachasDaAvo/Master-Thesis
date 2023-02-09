import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm import tqdm


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
                               pretrained=False if model_path else True)
        model.classifier.fc = nn.Linear(model.classifier.fc.in_features, 4)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        model = model.to(self.device).eval()
        return model

    def _get_outputs(self, images):
        input_batch = torch.stack(images, dim=0).to(self.device)

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


def save_tsne_results(tsne_results, labels, predicted_labels, probabilities, save_path, *, title=""):
    plt.figure(figsize=(16, 10))

    # predictions = [i == j for i, j in zip(labels, predicted_labels)]

    df = pd.DataFrame(
        {"x": tsne_results[:, 0], "y": tsne_results[:, 1],
         "labels": labels})  # , "predictions": predictions, "probabilities": probabilities})
    df["labels"] = df["labels"].map({0: "Covid", 1: "Lung Opacity", 2: "Normal", 3: "Viral Pneumonia"})
    # df["predictions"] = df["predictions"].map({True: "Correct", False: "Incorrect"})
    # df["probabilities"] = 1 - df["probabilities"]

    sns.scatterplot(data=df,
                    x="x",
                    y="y",
                    hue="labels",
                    # style="predictions",
                    # size="probabilities",
                    palette=sns.color_palette("hls", 4),
                    sizes=(10, 60),
                    alpha=0.5,
                    legend="brief")
    plt.legend(fontsize=15, loc="upper right")
    plt.title(title, fontsize=25)
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(save_path)


def load_fake_images():
    source_folder = ""
    fake_images = []

    for label in os.listdir(source_folder):
        count = 0
        for img_index in os.listdir(os.path.join(source_folder, label)):
            if count >= number_fake_images:
                break
            fake_images.append(Image.open(os.path.join(source_folder, label, img_index)).convert("RGB"))
            count += 1

    return fake_images


def classify_images(images_to_classify):
    # Build model
    test_model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b4", pretrained=False)
    test_model.classifier.fc = nn.Linear(test_model.classifier.fc.in_features, 4)
    test_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    test_model = test_model.to(device).eval()

    # Predict
    batched_images = [images_to_classify[i:i + 4] for i in range(0, len(images_to_classify), 4)]
    predictions = []
    probabilities = []
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    for images in tqdm(batched_images):
        input_batch = torch.stack([preprocess(image) for image in images], dim=0).to(device)
        with torch.no_grad():
            outputs = nn.functional.softmax(test_model(input_batch), dim=1)
            probs, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())

    return np.array(predictions), np.array(probabilities)


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


fake_features = load_pickle("")
fake_labels = load_pickle("")
tsne = TSNE(random_state=42, learning_rate="auto", init="pca", n_jobs=8)
fake_tsne_results = tsne.fit_transform(fake_features)

save_tsne_results(fake_tsne_results, fake_labels, None, None, "",
                  title="TSNE of fake images")

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = ""

# ["ACGAN", "ADCGAN", "BigGAN", "ContraGAN", "cStyleGAN2", "MHGAN", "ReACGAN"]
feature_extractor = FeatureExtractor(device, model_path)
tsne = TSNE(random_state=42, learning_rate="auto", init="pca", n_jobs=8)

data_path = ""
real_images = []
real_labels = []
for img_name, label in ImageFolder(data_path).imgs:
    real_images.append(transforms.ToTensor()(Image.open(img_name).convert("RGB")))
    real_labels.append(label)


real_features = feature_extractor.get_features(real_images)
fake_tsne_results = tsne.fit_transform(real_features)
save_tsne_results(fake_tsne_results,
                  real_labels,
                  None,
                  None,
                  f"",
                  title="TSNE of real images")

exit(0)

for gan in ["ACGAN", "ADCGAN", "BigGAN", "ContraGAN", "cStyleGAN2", "MHGAN", "ReACGAN"]:

    print(f"Starting {gan}")

    fake_images = []
    fake_labels = []
    for img_name, label in ImageFolder(f"").imgs:
        fake_images.append(transforms.ToTensor()(Image.open(img_name).convert("RGB")))
        fake_labels.append(label)

    fake_features = feature_extractor.get_features(fake_images)

    fake_tsne_results = tsne.fit_transform(fake_features)
    save_tsne_results(fake_tsne_results,
                      fake_labels,
                      None,
                      None,
                      f"",
                      title="TSNE of fake images")
"""
