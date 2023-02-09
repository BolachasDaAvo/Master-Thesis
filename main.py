import argparse
import os.path
import random
from collections import namedtuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.metrics as metrics
import torch
import torch.utils.data as torch_data
import torchvision.transforms as tfs
import wandb
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import augs.autoaug.aug as autoaug
import augs.baseline.aug as baseline
import augs.customrandaug.aug as customrandaug
import augs.ganaug.aug
import augs.ganaug.aug as ganaug
import augs.mixedaug.aug as mixedaug
import augs.randaug.aug as randaug
import augs.transform.aug as transform
import augs.trivialaugwide.aug as trivialaugwide


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
            img = t(img, label) if isinstance(t, (augs.ganaug.aug.Augment, augs.mixedaug.aug.Augment)) else t(img)

        return img, label

    def __len__(self):
        return len(self.data)


class GANAugDataset(torch_data.Dataset):
    def __init__(self, data_path, *, augmenter=None):
        self.data = []
        self.labels = []
        self.transform_list = []

        # Load data in memory
        for img_name, label in ImageFolder(data_path).imgs:
            self.data.append(tfs.ToTensor()(Image.open(img_name).convert(mode="RGB")))
            self.labels.append(label)

        # Build transforms
        if augmenter:
            self.transform_list.append(augmenter)

    def __getitem__(self, index):
        img = self.data[index]
        label = int(self.labels[index])

        for t in self.transform_list:
            img = t(img, label) if isinstance(t, (augs.ganaug.aug.Augment, augs.mixedaug.aug.Augment)) else t(img)

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

        if isinstance(augs, augs.ganaug.aug.Augment):
            self.train_dataset = GANAugDataset(os.path.join(self.data_paths[self.current_index], "train"),
                                               augmenter=self.augmenter)
        else:
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

            if aug_type in ["ganaug", "mixedaug"]:
                augmenter.next_gan()

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
    parser.add_argument("-aug_type", type=str)
    parser.add_argument("--N", type=int)
    parser.add_argument("--M", type=int)
    parser.add_argument("--transform", type=str)
    parser.add_argument("--policy", type=str)
    parser.add_argument("--gan", type=str)
    parser.add_argument("--selector", type=str)
    parser.add_argument("--p", type=float)
    parser.add_argument("--gan_checkpoint", nargs="+")
    parser.add_argument("--gan_config", nargs="+")

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
    aug_type = args.aug_type

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_file = open(log_path, "w")
    labels_readable = ["covid", "lung_opacity", "normal", "viral_pneumonia"]

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    augmenter = None
    if aug_type == "baseline":
        augmenter = baseline.Augment()
    elif aug_type == "transform":
        augmenter = transform.Augment(args.transform, args.M)
    elif aug_type == "randaug":
        augmenter = randaug.Augment(args.N, args.M)
    elif aug_type == "customrandaug":
        augmenter = customrandaug.Augment(args.N, args.M)
    elif aug_type == "trivialaugwide":
        augmenter = trivialaugwide.Augment()
    elif aug_type == "autoaug":
        augmenter = autoaug.Augment(args.policy)
    elif aug_type == "ganaug":
        if args.gan in ["FastGAN"]:
            args.gan_config = np.array(args.gan_config).reshape((num_splits, -1)).tolist()
            args.gan_checkpoint = np.array(args.gan_checkpoint).reshape((num_splits, -1)).tolist()
            mock_data = None
        else:
            mock_data = MultipleDatasetManager(data_paths, batch_size, num_workers, baseline.Augment())
        augmenter = ganaug.Augment(args.p, args.gan, args.selector, mock_data, args.gan_config, args.gan_checkpoint,
                                   device)
    elif aug_type == "mixedaug":
        if args.gan in ["FastGAN"]:
            args.gan_config = np.array(args.gan_config).reshape((num_splits, -1)).tolist()
            args.gan_checkpoint = np.array(args.gan_checkpoint).reshape((num_splits, -1)).tolist()
        augmenter = mixedaug.Augment(args.N, args.M, args.p, args.gan, args.gan_config, args.gan_checkpoint, device)
    else:
        raise Exception("Unknown augmentation")

    data = MultipleDatasetManager(data_paths, batch_size, num_workers, augmenter)
    rng = RngManager([i for i in range(42, 42 + num_runs)])

    train()
