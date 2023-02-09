import os

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

source = ""
columns = ["epoch", "val_accuracy", "val_covid_recall", "val_lung_opacity_recall", "val_normal_recall",
           "val_viral_pneumonia_recall", "val_covid_precision", "val_lung_opacity_precision", "val_normal_precision",
           "val_viral_pneumonia_precision"]
num_splits = 5
num_epochs = 70
top_exp = []
top_val = []
top_std = []


def get_average(df):
    df = df[columns]
    metrics = pd.DataFrame(columns=columns)
    for i in range(num_splits):
        best_epoch = df.iloc[[i for i in range(i * num_epochs, (i + 1) * num_epochs)]]
        best_epoch = best_epoch.iloc[[best_epoch["val_accuracy"].idxmax() - i * num_epochs]]
        metrics = pd.concat([metrics, best_epoch], ignore_index=True)

    s = ""
    for column in columns:
        s += f"{column}: {metrics[column].mean() * 100:.2f}+/-{metrics[column].std() * 100:.2f},"
        # s += f"{metrics[column].mean() * 100:.2f} & "
    print(s)

    top_val.append(metrics["val_accuracy"].mean())
    top_std.append(metrics["val_accuracy"].std())
    top_exp.append(experiment)


valid_transforms = ["shear_x", "shear_y", "translate_x", "translate_y", "rotate",
                    "brightness", "contrast", "sharpness", "posterize", "solarize",
                    "auto_contrast", "equalize", "perspective", "horizontal_flip", "vertical_flip",
                    "gaussian_blur"]
"""
for experiment in os.listdir(source):
    for transform in valid_transforms:
        if not experiment.startswith(transform) or "balance" in experiment:
            continue

        print(f"Experiment {experiment}")
        metrics_path = os.path.join(os.path.join(source, experiment), experiment + ".csv")
        try:
            data = pd.read_csv(metrics_path)
        except:
            print("no file")
            continue
        get_average(data)
"""

for experiment in os.listdir(source):
    if "realism" in experiment or "customrandaug" in experiment or "mixedaug" in experiment or "trivial" in experiment or \
            experiment.startswith("aug_") or "cls_" in experiment or experiment.startswith("ganaug_balance") or \
            "M3" in experiment or "M11" in experiment or "M7" in experiment:
        continue
    metrics_path = os.path.join(os.path.join(source, experiment), experiment + ".csv")
    try:
        data = pd.read_csv(metrics_path)
    except:
        print("no file")
        continue
    get_average(data)

print("--------------------------------------")
print(f"Sorted")
print("--------------------------------------")
high_scores = list(zip(top_exp, top_val, top_std))
high_scores.sort(reverse=True, key=lambda x: x[1])
for exp, val, std in high_scores:
    print(f"{exp}: {val * 100:.2f}+/-{std * 100:.2f}")
