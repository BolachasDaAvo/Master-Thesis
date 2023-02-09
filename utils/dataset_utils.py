import filecmp
import os
import random
import shutil

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import ImageFolder
from tqdm import tqdm

root = ""
folder_name = "Radiography"
source = ""

target_size = 64


def resize():
    new_dir = os.path.join(root, folder_name + str(target_size))
    os.mkdir(new_dir)

    for cls in os.listdir(source):
        new_item_dir = os.path.join(new_dir, cls)
        os.mkdir(new_item_dir)
        cls = os.path.join(source, cls)
        if not os.path.isdir(cls):
            continue
        for item in tqdm(os.listdir(cls)):
            item_path = os.path.join(cls, item)
            if not os.path.isfile(item_path):
                continue
            img = Image.open(item_path)
            resized_img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            resized_img.save(os.path.join(new_item_dir, item), "PNG")


def train_test_split():
    source = ""
    dest = ""

    train_folder = os.path.join(source)
    valid_folder = os.path.join(dest)

    for cls in os.listdir(source):
        new_cls_dir = os.path.join(valid_folder, cls)
        os.mkdir(new_cls_dir)
        cls = os.path.join(source, cls)

        files = os.listdir(cls)
        print(f"Found {len(files)} images for class {cls}")
        n_files = len(files) // 5
        print(f"Selecting {n_files} random images")

        for file_name in random.sample(files, n_files):
            shutil.move(os.path.join(cls, file_name), new_cls_dir)


def check():
    root = ""

    classes = {"covid": "COVID", "lung_opacity": "Lung_Opacity", "normal": "Normal",
               "viral_pneumonia": "Viral Pneumonia"}
    for cls in classes.keys():
        _dir = os.path.join(root, cls)
        files = os.listdir(_dir)

        for file_name in files:
            if not file_name.startswith(classes[cls]):
                print("error", file_name)


def split_with_indices():
    def mkdir(name):
        try:
            os.mkdir(name)
        except:
            pass

    def set_rng(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def get_indices(num_splits, file_names, labels, seed):
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
        return list(skf.split(file_names, labels))

    source = ""
    root = ""
    class_mapping = {0: "covid", 1: "lung_opacity", 2: "normal", 3: "viral_pneumonia"}
    num_splits = 5
    seed = 42
    set_rng(seed)

    # get split file names
    data = ImageFolder(root=source)
    file_names = list(map(lambda x: x[0], data.imgs))
    labels = list(map(lambda x: x[1], data.imgs))
    indices = get_indices(num_splits, file_names, labels, seed)

    for i, (train_indices, test_indices) in enumerate(indices):

        # Create dirs
        dest = f"{root}_{i + 1}"
        mkdir(dest)
        mkdir(os.path.join(dest, "train"))
        mkdir(os.path.join(dest, "valid"))
        for cls in class_mapping.values():
            mkdir(os.path.join(dest, "train", cls))
            mkdir(os.path.join(dest, "valid", cls))

        # Train
        for idx in train_indices:
            shutil.copy(file_names[idx], os.path.join(dest, "train", class_mapping[labels[idx]]))

        for idx in test_indices:
            shutil.copy(file_names[idx], os.path.join(dest, "valid", class_mapping[labels[idx]]))


def check_equal():
    _1 = ""
    _2 = ""

    def are_dir_trees_equal(dir1, dir2):
        dirs_cmp = filecmp.dircmp(dir1, dir2)
        if len(dirs_cmp.left_only) > 0 or len(dirs_cmp.right_only) > 0 or \
                len(dirs_cmp.funny_files) > 0:
            return False
        (_, mismatch, errors) = filecmp.cmpfiles(
            dir1, dir2, dirs_cmp.common_files, shallow=False)
        if len(mismatch) > 0 or len(errors) > 0:
            return False
        for common_dir in dirs_cmp.common_dirs:
            new_dir1 = os.path.join(dir1, common_dir)
            new_dir2 = os.path.join(dir2, common_dir)
            if not are_dir_trees_equal(new_dir1, new_dir2):
                return False
        return True

    print(are_dir_trees_equal(_1, _2))


def convert_to_rgb():
    source_dir = ""
    dest_dir = ""

    for dataset_split in os.listdir(source_dir):
        os.mkdir(os.path.join(dest_dir, dataset_split))
        for train_test in os.listdir(os.path.join(source_dir, dataset_split)):
            os.mkdir(os.path.join(dest_dir, dataset_split, train_test))
            for cls in os.listdir(os.path.join(source_dir, dataset_split, train_test)):
                os.mkdir(os.path.join(dest_dir, dataset_split, train_test, cls))
                for greyscale_img in os.listdir(os.path.join(source_dir, dataset_split, train_test, cls)):
                    rgb_image = Image.open(os.path.join(source_dir, dataset_split, train_test, cls, greyscale_img)).convert("RGB")
                    rgb_image.save(os.path.join(dest_dir, dataset_split, train_test, cls, greyscale_img))
                    rgb_image.close()


convert_to_rgb()
