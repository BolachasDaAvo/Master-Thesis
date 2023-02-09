import os
import random
import shutil

save_root = ""

dataset_paths = []

dataset_names = []

ratios = [5, 10, 15, 20]

number_to_name = {0: "covid", 1: "lung_opacity", 2: "normal", 3: "viral_pneumonia"}

for dataset_path, dataset_name in zip(dataset_paths, dataset_names):
    for ratio in ratios:
        new_folder_path = os.path.join(dataset_path, f"{dataset_name}_r{ratio}")

        already_moved = []
        for cls in range(4):
            cls_path = os.path.join(new_folder_path, "train", number_to_name[cls])

            images = os.listdir(cls_path)
            print(f"Found {len(images)} images for class {cls}")
            n_images = int(len(images) * (ratio / 100))
            print(f"Selecting {n_images} random images")

            for img in already_moved:
                try:
                    images.remove(img)
                except Exception:
                    pass
            random_sample = random.sample(images, n_images)

            for img in random_sample:
                new_cls = random.randint(0, 3)
                while new_cls == cls:
                    new_cls = random.randint(0, 3)
                shutil.move(os.path.join(cls_path, img),
                            os.path.join(new_folder_path, "train", number_to_name[new_cls]))
                already_moved.append(img)
