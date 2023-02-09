import os

import pandas as pd

source = ""
columns = ["val_accuracy", "val_covid_recall", "val_lung_opacity_recall", "val_normal_recall",
           "val_viral_pneumonia_recall"]
num_splits = 5
num_epochs = 70


def average_metrics(df):
    df = df[columns]
    metrics = pd.DataFrame(columns=columns)
    for i in range(num_splits):
        best_epoch = df.iloc[[i for i in range(i * num_epochs, (i + 1) * num_epochs)]]
        best_epoch = best_epoch.iloc[[best_epoch["val_accuracy"].idxmax() - i * num_epochs]]
        metrics = pd.concat([metrics, best_epoch], ignore_index=True)

    return metrics


def autoaug_balance():
    valid_dataset = {"cifar10": 0, "imagenet": 1, "svhn": 2}
    valid_t = {100: 0, 200: 1, 300: 2, 400: 3, 600: 4, 800: 5, 1000: 6}
    readable_map = {"cifar10": "Cifar10", "imagenet": "Imagenet", "svhn": "SVHN"}

    table = []
    for t in range(len(valid_t.keys())):
        table.append([])
        for d in range(len(valid_dataset.keys())):
            table[t].append([""])

    search_string = "autoaug_balance"

    # Get files
    for experiment in os.listdir(source):
        if search_string in experiment:
            # Get d and t
            d = experiment[experiment.rfind("_") + 1:len(experiment)]
            t = int(experiment[len(search_string) + 1:experiment.find("_", len(search_string) + 1, len(experiment))])
            print(d, t)

            if d not in valid_dataset.keys() or t not in valid_t.keys():
                print(f"skipping {experiment}")
                continue
            print(experiment)

            metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
            averaged_metrics = average_metrics(metrics)

            table[valid_t[t]][valid_dataset[d]][0] = f"{averaged_metrics['val_accuracy'].mean() * 100:.2f}"

    table_string = ""
    # Deconstruct table
    for t in sorted(valid_t.values()):
        for d in sorted(valid_dataset.keys()):
            table_string += f" & {readable_map[d]} & {table[t][valid_dataset[d]][0]} \\\\\n"

    print(table_string)


def transform_balance():
    valid_M = {1: 0, 7: 1, 13: 2}
    valid_t = {100: 0, 200: 1, 300: 2, 400: 3, 600: 4, 800: 5, 1000: 6}
    valid_transforms = {"shear_x": 10, "shear_y": 11, "translate_x": 13, "translate_y": 14, "rotate": 8,
                        "brightness": 1, "contrast": 2, "sharpness": 9, "posterize": 7, "solarize": 12,
                        "auto_contrast": 0, "equalize": 3, "perspective": 6, "horizontal_flip": 5, "vertical_flip": 15,
                        "gaussian_blur": 4}

    readable_map = {"shear_x": "Horizontal Shear", "shear_y": "Vertical Shear", "translate_x": "Horizontal Translation",
                    "translate_y": "Vertical Translation", "rotate": "Rotate", "brightness": "Brightness",
                    "contrast": "Contrast", "sharpness": "Sharpness", "posterize": "Posterize", "solarize": "Solarize",
                    "auto_contrast": "Auto Contrast", "equalize": "Equalize", "perspective": "Perspective",
                    "horizontal_flip": "Horizontal Flip", "vertical_flip": "Vertical Flip",
                    "gaussian_blur": "Gaussian Blur"}

    table = []
    for t in range(len(valid_t.keys())):
        table.append([])
        for transform in range(len(valid_transforms.keys())):
            table[t].append([])
            for M in range(len(valid_M.keys())):
                table[t][transform].append("")

    search_string = "_balance"

    # Get files
    for experiment in os.listdir(source):
        for transform in valid_transforms.keys():
            if not experiment.startswith(transform + search_string):
                continue
            print(transform)
            print(experiment)

            # Get N, M, and t
            M = int(experiment[experiment.find("M") + 1:len(experiment)])
            t = int(experiment[
                    len(transform + search_string + "_"):experiment.find("_",
                                                                         len(transform + search_string + "_"),
                                                                         len(experiment))])
            print(t, transform, M)

            if M not in valid_M.keys() or t not in valid_t.keys():
                print(f"skipping {experiment}")
                continue

            metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
            averaged_metrics = average_metrics(metrics)

            table[valid_t[t]][valid_transforms[transform]][
                valid_M[M]] = f"{averaged_metrics['val_accuracy'].mean() * 100:.2f}"

    table_string = ""
    # Deconstruct table
    for t in sorted(valid_t.values()):
        table_string += "-----------------------------------------------------------\n"
        for transform in sorted(valid_transforms.keys()):
            line = f" & {readable_map[transform]}"
            for M in sorted(valid_M.values()):
                line += f" & {table[t][valid_transforms[transform]][M]}"
            line += " \\\\\n"
            table_string += line

    print(table_string)


def randaug_balance():
    valid_M = {1: 0, 5: 1, 9: 2, 13: 3}
    valid_N = {1: 0, 2: 1, 3: 2, 4: 3}
    valid_t = {100: 0, 200: 1, 300: 2, 400: 3, 600: 4, 800: 5, 1000: 6}

    table = []
    for t in range(len(valid_t.keys())):
        table.append([])
        for N in range(len(valid_N.keys())):
            table[t].append([])
            for M in range(len(valid_M.keys())):
                table[t][N].append("")

    search_string = "randaug_balance"

    # Get files
    for experiment in os.listdir(source):
        if search_string in experiment:
            # Get N, M, and t
            N = int(experiment[experiment.find("N") + 1])
            M = int(experiment[experiment.find("M") + 1:len(experiment)])
            t = int(experiment[16:experiment.find("_", 16, len(experiment))])

            if N not in valid_N.keys() or M not in valid_M.keys() or t not in valid_t.keys():
                print(f"skipping {experiment}")
                continue
            print(experiment)

            metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
            averaged_metrics = average_metrics(metrics)

            table[valid_t[t]][valid_N[N]][valid_M[M]] = f"{averaged_metrics['val_accuracy'].mean() * 100:.2f}"

    table_string = ""
    # Deconstruct table
    for t in sorted(valid_t.values()):
        for N in sorted(valid_N.values()):
            line = f" & {N}"
            for M in sorted(valid_M.values()):
                line += f" & {table[t][N][M]}"
            line += " \\\\\n"
            table_string += line

    print(table_string)


def ganaug_balance():
    valid_arq = {"ACGAN": 0, "ADCGAN": 1, "BigGAN": 2, "ContraGAN": 3, "cStyleGAN2": 4, "MHGAN": 5, "ReACGAN": 6}
    valid_t = {100: 0, 200: 1, 300: 2, 400: 3, 600: 4, 800: 5, 1000: 6}

    table = []
    for t in range(len(valid_t.keys())):
        table.append([])
        for a in range(len(valid_arq.keys())):
            table[t].append([""])

    search_string = "ganaug_with_normalization_balance_"

    # Get files
    for experiment in os.listdir(source):
        if search_string in experiment:
            # Get t and a
            t = int(experiment[len(search_string): experiment.find("_", len(search_string), len(experiment))])
            for arq in valid_arq:
                if arq in experiment:
                    if arq == "ACGAN" and "ReACGAN" in experiment:
                        continue
                    a = arq
                    break
            print(t, a)

            if a not in valid_arq.keys() or t not in valid_t.keys():
                print(f"skipping {experiment}")
                continue
            print(experiment)

            metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
            averaged_metrics = average_metrics(metrics)

            table[valid_t[t]][valid_arq[a]][0] = f"{averaged_metrics['val_accuracy'].mean() * 100:.2f}"

    table_string = ""
    # Deconstruct table
    for t in sorted(valid_t.values()):
        for a in sorted(valid_arq.keys()):
            table_string += f" & {a} & {table[t][valid_arq[a]][0]} \\\\\n"

    print(table_string)


ganaug_balance()
