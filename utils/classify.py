import glob
import os
import statistics
from collections import namedtuple

import numpy as np
import sklearn.metrics as skl_metrics
import torch
import torch.nn as nn

from augs.ganaug.generators.studiogan.gen import ImageGenerator
from augs.ganaug.selectors.identity import Selector

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels_readable = ["covid", "lung_opacity", "normal", "viral_pneumonia"]
batch_size = 32

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
         "ProjGAN": Checkpoint(gan="ProjGAN",
                               root=""),
         "ReACGAN": Checkpoint(gan="ReACGAN",
                               root=""),
         "cStyleGAN2": Checkpoint(gan="cStyleGAN2",
                                  root=""),
         }

model_paths = []


def get_metrics(y_true, y_pred):
    report_metrics = {}

    report = skl_metrics.classification_report(y_true, y_pred, labels=list(range(4)), target_names=labels_readable,
                                               output_dict=True)

    names = ["recall"]
    for label in labels_readable:
        for n in names:
            report_metrics[f"{label}_{n}"] = report[label][n]
    report_metrics[f"accuracy"] = report["accuracy"]

    return report_metrics


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
        checkpoint_paths.append(split)

    config_paths = sorted(config_paths)
    checkpoint_paths = sorted(checkpoint_paths)

    aggregated_metrics = {"accuracy": []}
    aggregated_metrics.update({f"{key}_recall": [] for key in labels_readable})

    y_pred = []
    y_prob = []
    for cfg_path, ckpt_path, model_path in zip(config_paths, checkpoint_paths, model_paths):

        # Fake images
        generator = ImageGenerator(cfg_path, ckpt_path, Selector(), device)
        fake_labels = [i for i in range(0, 4)] * 2500
        print("Generating images...")
        fake_images = [generator.forward(label) for label in fake_labels]
        fake_images = [fake_images[i:i + batch_size] for i in range(0, len(fake_images), batch_size)]

        # Model
        test_model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b4", pretrained=False)
        test_model.classifier.fc = nn.Linear(test_model.classifier.fc.in_features, 4)
        test_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        test_model = test_model.to(device).eval()

        # Predict
        print("Classifying images...")
        running_y_true = fake_labels
        running_y_pred = []
        running_y_prob = []
        for images in fake_images:
            input_batch = torch.stack(images, dim=0).to(device)
            with torch.no_grad():
                probabilities = nn.functional.softmax(test_model(input_batch), dim=1)
                prob, preds = torch.max(probabilities, 1)

            y_prob.extend(prob.squeeze().tolist())
            y_pred.extend(preds.cpu().tolist())
            running_y_pred.extend(preds.cpu().tolist())

        metrics = get_metrics(running_y_true, running_y_pred)
        for key, value in metrics.items():
            aggregated_metrics[key] += [value]

    # Probs
    for label in range(0, 4):
        y_prob_label = [y_prob[i] for i in y_pred if i == label]
        bins = np.histogram(np.array(y_prob_label), bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        print(bins[0] / 2500 * 100)

    # Metrics
    for key, value in aggregated_metrics.items():
        aggregated_metrics[key] = statistics.mean(aggregated_metrics[key])
    print(aggregated_metrics)
