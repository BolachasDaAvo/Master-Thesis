import itertools
import os
import shutil
import warnings

import matplotlib.pyplot as plt
import pandas
import pandas as pd
import seaborn as sns

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def cgans():
    def rename_files():
        for gan in gans:
            files_path = f""
            files = sorted(os.listdir(files_path))
            for file, new_name in zip(files, metrics):
                shutil.copy(os.path.join(files_path, file), os.path.join(files_path, f"{new_name}.csv"))

    max_step = 100000
    gans = ["ACGAN", "ADCGAN", "BigGAN", "ContraGAN", "cStyleGAN2", "MHGAN", "ProjGAN", "ReACGAN"]
    metrics = ["FID", "IS", "Improved Precision", "Improved Recall"]
    # Rename files
    # rename_files()

    for gan, data_name in itertools.product(gans, metrics):

        data_names = {"FID": (0, 400 if gan in ["ACGAN", "ProjGAN"] else 250), "IS": (0, 3),
                      "Improved Precision": (0, 1), "Improved Recall": (0, 1)}
        root = f""
        graph_name = f"{gan} - {data_name}"

        try:
            data = pd.read_csv(f"{root}/{data_name}.csv")
        except Exception as e:
            print(e)
            continue
        column_names = data.columns

        # Drop irrelevant columns
        data = data.drop([column for column in column_names if "MAX" in column or "MIN" in column], axis=1)

        # Sort columns
        data = data.sort_index(axis=1)

        # Rename columns
        def renamer(name):
            if "_1-train" in name:
                return "Data fold 1"
            elif "_2-train" in name:
                return "Data fold 2"
            elif "_3-train" in name:
                return "Data fold 3"
            elif "_4-train" in name:
                return "Data fold 4"
            elif "_5-train" in name:
                return "Data fold 5"
            elif "Step" in name:
                return "Epoch"
            else:
                return name

        data = data.rename(renamer, axis=1)

        # Remove steps
        data = data.drop(data[data.Epoch > max_step].index)

        # Rearrange data
        data = pd.melt(data, ["Epoch"], var_name="Data Fold", value_name=data_name)

        # Draw plot
        sns.lineplot(data=data,
                     x="Epoch",
                     y=data_name,
                     hue="Data Fold",
                     palette=sns.color_palette("hls", 5),
                     linewidth=1.5)
        plt.legend(fontsize=10, loc="upper right")
        plt.title(graph_name, fontsize=15, pad=25)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().grid(axis="y", color="lightgrey")
        plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10, width=1)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["bottom"].set_color("lightgrey")
        plt.gca().set_ylim(data_names[data_name])
        plt.gca().set_xlim(0, max_step)
        plt.xlabel("Epoch", labelpad=8)
        plt.ylabel(data_name, labelpad=8)

        def formatter(label, pos):
            label = str(int(label))
            if int(label) < 1000:
                return label
            reduced_label = label[:-3]
            return reduced_label + "k"

        plt.gca().xaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plt.savefig(f"{root}/{graph_name}.png")
        plt.close()


def classifiers():
    def average_metrics(df):
        df = df[columns]
        metrics = pd.DataFrame(columns=columns)
        for i in range(num_splits):
            best_epoch = df.iloc[[i for i in range(i * num_epochs, (i + 1) * num_epochs)]]
            best_epoch = best_epoch.iloc[[best_epoch["val_accuracy"].idxmax() - i * num_epochs]]
            metrics = pd.concat([metrics, best_epoch], ignore_index=True)

        return metrics

    def transform_balance():
        valid_M = {1: 0, 7: 1, 13: 2}
        valid_t = {100: 0, 200: 1, 300: 2, 400: 3, 600: 4, 800: 5, 1000: 6}
        valid_transforms = {"shear_x": 10, "shear_y": 11, "translate_x": 13, "translate_y": 14, "rotate": 8,
                            "brightness": 1, "contrast": 2, "sharpness": 9, "posterize": 7, "solarize": 12,
                            "auto_contrast": 0, "equalize": 3, "perspective": 6, "horizontal_flip": 5,
                            "vertical_flip": 15,
                            "gaussian_blur": 4}

        readable_map = {"shear_x": "Horizontal Shear", "shear_y": "Vertical Shear",
                        "translate_x": "Horizontal Translation",
                        "translate_y": "Vertical Translation", "rotate": "Rotate", "brightness": "Brightness",
                        "contrast": "Contrast", "sharpness": "Sharpness", "posterize": "Posterize",
                        "solarize": "Solarize",
                        "auto_contrast": "Auto Contrast", "equalize": "Equalize", "perspective": "Perspective",
                        "horizontal_flip": "Horizontal Flip", "vertical_flip": "Vertical Flip",
                        "gaussian_blur": "Gaussian Blur"}

        """
        search_string = "_balance"
        aggregated_metrics = pd.DataFrame(columns=["t", "transform", "M", *columns])
        # Get files
        for experiment in os.listdir(source):
            for transform in valid_transforms.keys():
                if not experiment.startswith(transform + search_string):
                    continue

                # Get M, and t
                M = int(experiment[experiment.find("M") + 1:len(experiment)])
                t = int(experiment[
                        len(transform + search_string + "_"):experiment.find("_", len(transform + search_string + "_"),
                                                                             len(experiment))])

                if M not in valid_M.keys() or t not in valid_t.keys():
                    continue

                metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
                averaged_metrics = average_metrics(metrics)
                new_row = {"t": t, "transform": readable_map[transform], "M": M}
                for column in columns:
                    new_row[column] = round(averaged_metrics[column].mean(), 4)
                aggregated_metrics = aggregated_metrics.append(new_row, ignore_index=True)

        aggregated_metrics = aggregated_metrics.rename(
            columns={"t": "Threshold", "transform": "Transformation", "M": "Magnitude", "val_accuracy": "Accuracy",
                     "val_covid_recall": "Covid Recall", "val_lung_opacity_recall": "Lung Opacity Recall",
                     "val_normal_recall": "Normal Recall", "val_viral_pneumonia_recall": "Viral Pneumonia Recall", })
        aggregated_metrics.to_csv(
            "", index=False)
        """

        aggregated_metrics = pd.read_csv(
            "")

        for transform in valid_transforms.keys():
            filtered_aggregated_metrics = aggregated_metrics[
                aggregated_metrics.Transformation == readable_map[transform]]

            # Lines
            sns.lineplot(data=filtered_aggregated_metrics,
                         x="Magnitude",
                         y="Accuracy",
                         hue="Threshold",
                         palette=sns.color_palette("ch:start=.2,rot=-.3", n_colors=len(valid_t.keys())),
                         legend="full",
                         linewidth=1,
                         alpha=0.5,
                         )

            # Trend Line
            sns.regplot(data=filtered_aggregated_metrics,
                        x="Magnitude",
                        y="Accuracy",
                        color="red",
                        scatter=False,
                        ci=None,
                        line_kws={"linestyle": "dashed", "linewidth": 2},
                        lowess=True,
                        )

            # Baseline
            plt.plot([1, 7, 13], [0.6979, 0.6979, 0.6979], color="green", linewidth=2, linestyle="dashed")

            # Legend
            plt.legend(fontsize=10, loc="upper right", title="Threshold (t)")

            # Format
            plt.title(f"{readable_map[transform]}", fontsize=15, pad=25)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().grid(axis="y", color="lightgrey")
            plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                                  width=1)
            plt.gca().spines["bottom"].set_linewidth(2)
            plt.gca().spines["bottom"].set_color("lightgrey")
            plt.gca().set_ylim((0.68, 0.79))
            plt.gca().set_xlim((1, 13))
            plt.gca().set_yticks([0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79])
            plt.gca().set_xticks(list(valid_M.keys()))
            plt.xlabel("Magnitude (M)", labelpad=8)
            plt.ylabel("Accuracy", labelpad=8)

            plt.tight_layout()
            plt.savefig(
                f"")
            plt.close()

        exit(0)

        #######################################################################
        #######################################################################
        #######################################################################

        best_aggregated_metrics = pandas.DataFrame(columns=aggregated_metrics.columns)
        for transform in valid_transforms.keys():
            for threshold in valid_t.keys():
                transform_metrics = aggregated_metrics[
                    (aggregated_metrics.Transformation == readable_map[transform]) & (
                            aggregated_metrics.Threshold == threshold)]
                best_aggregated_metrics = best_aggregated_metrics.append(
                    aggregated_metrics.loc[[transform_metrics["Accuracy"].idxmax()]])

        # Lines
        sns.lineplot(data=best_aggregated_metrics,
                     x="Threshold",
                     y="Accuracy",
                     hue="Transformation",
                     palette=sns.color_palette("hls", len(valid_transforms.keys())),
                     legend="full",
                     linewidth=1,
                     alpha=0.8,
                     )

        # Trend Line
        best_aggregated_metrics.Threshold = pd.to_numeric(best_aggregated_metrics.Threshold)
        best_aggregated_metrics.Accuracy = pd.to_numeric(best_aggregated_metrics.Accuracy)
        sns.regplot(data=best_aggregated_metrics,
                    x="Threshold",
                    y="Accuracy",
                    color="red",
                    scatter=False,
                    ci=None,
                    line_kws={"linestyle": "dashed", "linewidth": 2},
                    lowess=True,
                    )

        # Baseline
        plt.plot([100, 200, 300, 400, 600, 800, 1000], [0.6979, 0.6979, 0.6979, 0.6979, 0.6979, 0.6979, 0.6979],
                 color="green", linewidth=2, linestyle="dashed")

        # Legend
        plt.legend(fontsize=6, loc="upper left", title="Transformation", ncol=4)

        # Format
        plt.title("Accuracy", fontsize=15, pad=25)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().grid(axis="y", color="lightgrey")
        plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                              width=1)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["bottom"].set_color("lightgrey")
        # plt.gca().set_ylim((0.68, 0.79))
        plt.gca().set_xlim((100, 1000))
        # plt.gca().set_yticks([0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79])
        plt.gca().set_xticks(list(valid_t.keys()))
        plt.xlabel("Threshold (t)", labelpad=8)
        plt.ylabel("Accuracy", labelpad=8)

        plt.tight_layout()
        plt.savefig(
            f"")
        plt.close()

        #####################################################################

        # Lines
        sns.lineplot(data=best_aggregated_metrics,
                     x="Threshold",
                     y="Covid Recall",
                     hue="Transformation",
                     palette=sns.color_palette("hls", len(valid_transforms.keys())),
                     legend="full",
                     linewidth=1,
                     alpha=0.8,
                     )

        # Trend Line
        best_aggregated_metrics.Threshold = pd.to_numeric(best_aggregated_metrics.Threshold)
        best_aggregated_metrics.Accuracy = pd.to_numeric(best_aggregated_metrics.Accuracy)
        sns.regplot(data=best_aggregated_metrics,
                    x="Threshold",
                    y="Covid Recall",
                    color="red",
                    scatter=False,
                    ci=None,
                    line_kws={"linestyle": "dashed", "linewidth": 2},
                    lowess=True,
                    )

        # Baseline
        plt.plot([100, 200, 300, 400, 600, 800, 1000], [0.4222, 0.4222, 0.4222, 0.4222, 0.4222, 0.4222, 0.4222],
                 color="green", linewidth=2, linestyle="dashed")

        # Legend
        plt.legend(fontsize=6, loc="upper left", title="Transformation", ncol=4)

        # Format
        plt.title("Covid Recall", fontsize=15, pad=25)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().grid(axis="y", color="lightgrey")
        plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                              width=1)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["bottom"].set_color("lightgrey")
        plt.gca().set_ylim((0.30, 0.70))
        plt.gca().set_xlim((100, 1000))
        # plt.gca().set_yticks([0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
        plt.gca().set_xticks(list(valid_t.keys()))
        plt.xlabel("Threshold (t)", labelpad=8)
        plt.ylabel("Covid Recall", labelpad=8)

        plt.tight_layout()
        plt.savefig(
            f"")
        plt.close()

        # Lines
        sns.lineplot(data=best_aggregated_metrics,
                     x="Threshold",
                     y="Lung Opacity Recall",
                     hue="Transformation",
                     palette=sns.color_palette("hls", len(valid_transforms.keys())),
                     legend="full",
                     linewidth=1,
                     alpha=0.8,
                     )

        # Trend Line
        best_aggregated_metrics.Threshold = pd.to_numeric(best_aggregated_metrics.Threshold)
        best_aggregated_metrics.Accuracy = pd.to_numeric(best_aggregated_metrics.Accuracy)
        sns.regplot(data=best_aggregated_metrics,
                    x="Threshold",
                    y="Lung Opacity Recall",
                    color="red",
                    scatter=False,
                    ci=None,
                    line_kws={"linestyle": "dashed", "linewidth": 2},
                    lowess=True,
                    )

        # Baseline
        plt.plot([100, 200, 300, 400, 600, 800, 1000], [0.6200, 0.6200, 0.6200, 0.6200, 0.6200, 0.6200, 0.6200],
                 color="green", linewidth=2, linestyle="dashed")

        # Legend
        plt.legend(fontsize=6, loc="upper left", title="Transformation", ncol=4)

        # Format
        plt.title("Lung Opacity Recall", fontsize=15, pad=25)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().grid(axis="y", color="lightgrey")
        plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                              width=1)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["bottom"].set_color("lightgrey")
        plt.gca().set_ylim((0.6, 0.80))
        plt.gca().set_xlim((100, 1000))
        # plt.gca().set_yticks([0.60, 0.65, 0.70, 0.75, 0.80])
        plt.gca().set_xticks(list(valid_t.keys()))
        plt.xlabel("Threshold (t)", labelpad=8)
        plt.ylabel("Lung Opacity Recall", labelpad=8)

        plt.tight_layout()
        plt.savefig(
            f"")
        plt.close()

        # Lines
        sns.lineplot(data=best_aggregated_metrics,
                     x="Threshold",
                     y="Normal Recall",
                     hue="Transformation",
                     palette=sns.color_palette("hls", len(valid_transforms.keys())),
                     legend="full",
                     linewidth=1,
                     alpha=0.8,
                     )

        # Trend Line
        best_aggregated_metrics.Threshold = pd.to_numeric(best_aggregated_metrics.Threshold)
        best_aggregated_metrics.Accuracy = pd.to_numeric(best_aggregated_metrics.Accuracy)
        sns.regplot(data=best_aggregated_metrics,
                    x="Threshold",
                    y="Normal Recall",
                    color="red",
                    scatter=False,
                    ci=None,
                    line_kws={"linestyle": "dashed", "linewidth": 2},
                    lowess=True,
                    )

        # Baseline
        plt.plot([100, 200, 300, 400, 600, 800, 1000], [0.8704, 0.8704, 0.8704, 0.8704, 0.8704, 0.8704, 0.8704],
                 color="green", linewidth=2, linestyle="dashed")

        # Legend
        plt.legend(fontsize=6, loc="upper left", title="Transformation", ncol=4)

        # Format
        plt.title("Normal Recall", fontsize=15, pad=25)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().grid(axis="y", color="lightgrey")
        plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                              width=1)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["bottom"].set_color("lightgrey")
        plt.gca().set_ylim((0.78, 0.90))
        plt.gca().set_xlim((100, 1000))
        # plt.gca().set_yticks([0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79])
        plt.gca().set_xticks(list(valid_t.keys()))
        plt.xlabel("Threshold (t)", labelpad=8)
        plt.ylabel("Normal Recall", labelpad=8)

        plt.tight_layout()
        plt.savefig(
            f"")
        plt.close()

        # Lines
        sns.lineplot(data=best_aggregated_metrics,
                     x="Threshold",
                     y="Viral Pneumonia Recall",
                     hue="Transformation",
                     palette=sns.color_palette("hls", len(valid_transforms.keys())),
                     legend="full",
                     linewidth=1,
                     alpha=0.8,
                     )

        # Trend Line
        best_aggregated_metrics.Threshold = pd.to_numeric(best_aggregated_metrics.Threshold)
        best_aggregated_metrics.Accuracy = pd.to_numeric(best_aggregated_metrics.Accuracy)
        sns.regplot(data=best_aggregated_metrics,
                    x="Threshold",
                    y="Viral Pneumonia Recall",
                    color="red",
                    scatter=False,
                    ci=None,
                    line_kws={"linestyle": "dashed", "linewidth": 2},
                    lowess=True,
                    )

        # Baseline
        plt.plot([100, 200, 300, 400, 600, 800, 1000], [0.4758, 0.4758, 0.4758, 0.4758, 0.4758, 0.4758, 0.4758],
                 color="green", linewidth=2, linestyle="dashed")

        # Legend
        plt.legend(fontsize=6, loc="upper left", title="Transformation", ncol=4)

        # Format
        plt.title("Viral Pneumonia Recall", fontsize=15, pad=25)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().grid(axis="y", color="lightgrey")
        plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                              width=1)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["bottom"].set_color("lightgrey")
        plt.gca().set_ylim((0.45, 0.90))
        plt.gca().set_xlim((100, 1000))
        # plt.gca().set_yticks([0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79])
        plt.gca().set_xticks(list(valid_t.keys()))
        plt.xlabel("Threshold (t)", labelpad=8)
        plt.ylabel("Viral Pneumonia Recall", labelpad=8)

        plt.tight_layout()
        plt.savefig(
            f"")
        plt.close()

    def autoaug_balance():
        valid_dataset = {"cifar10": 0, "imagenet": 1, "svhn": 2}
        valid_t = {100: 0, 200: 1, 300: 2, 400: 3, 600: 4, 800: 5, 1000: 6}
        readable_map = {"cifar10": "Cifar10", "imagenet": "Imagenet", "svhn": "SVHN"}

        search_string = "autoaug_balance"
        aggregated_metrics = pd.DataFrame(columns=["t", "Policy", *columns])

        # Get files
        for experiment in os.listdir(source):
            if search_string in experiment:
                # Get d and t
                d = experiment[experiment.rfind("_") + 1:len(experiment)]
                t = int(
                    experiment[len(search_string) + 1:experiment.find("_", len(search_string) + 1, len(experiment))])

                if d not in valid_dataset.keys() or t not in valid_t.keys():
                    continue

                metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
                averaged_metrics = average_metrics(metrics)
                new_row = {"t": t, "Policy": readable_map[d]}
                for column in columns:
                    new_row[column] = round(averaged_metrics[column].mean(), 4)
                aggregated_metrics = aggregated_metrics.append(new_row, ignore_index=True)

        aggregated_metrics = aggregated_metrics.rename(
            columns={"t": "Threshold", "Policy": "Policy", "val_accuracy": "Accuracy",
                     "val_covid_recall": "Covid Recall", "val_lung_opacity_recall": "Lung Opacity Recall",
                     "val_normal_recall": "Normal Recall", "val_viral_pneumonia_recall": "Viral Pneumonia Recall", })
        aggregated_metrics.to_csv(
            "", index=False)

        aggregated_metrics = pd.read_csv(
            "")

        metric_map = {"val_accuracy": "Accuracy", "val_covid_recall": "Covid Recall",
                      "val_lung_opacity_recall": "Lung Opacity Recall", "val_normal_recall": "Normal Recall",
                      "val_viral_pneumonia_recall": "Viral Pneumonia Recall"}
        metric_map = {v: k for k, v in metric_map.items()}
        baselines = {"Accuracy": 0.6979, "Covid Recall": 0.4222, "Lung Opacity Recall": 0.6200, "Normal Recall": 0.8704,
                     "Viral Pneumonia Recall": 0.4758}
        y_limits = {"Accuracy": (0.68, 0.8), "Covid Recall": (0.4, 0.7), "Lung Opacity Recall": (0.6, 0.8),
                    "Normal Recall": (0.70, 0.90), "Viral Pneumonia Recall": (0.4, 0.9)}
        x_limits = {"Accuracy": (100, 1000), "Covid Recall": (100, 1000), "Lung Opacity Recall": (100, 1000),
                    "Normal Recall": (100, 1000), "Viral Pneumonia Recall": (100, 1000)}
        for metric in aggregated_metrics.columns[-5:]:
            # Lines
            sns.lineplot(data=aggregated_metrics,
                         x="Threshold",
                         y=metric,
                         hue="Policy",
                         palette=sns.color_palette("hls", len(valid_dataset.keys())),
                         legend="full",
                         linewidth=1,
                         alpha=0.8,
                         )

            sns.regplot(data=aggregated_metrics,
                        x="Threshold",
                        y=metric,
                        color="red",
                        scatter=False,
                        ci=None,
                        line_kws={"linestyle": "dashed", "linewidth": 2},
                        lowess=True,
                        )

            # Baseline
            plt.plot(valid_t.keys(),
                     [baselines[metric]] * len(valid_t.keys()),
                     color="green",
                     linewidth=2,
                     linestyle="dashed")

            # Legend
            plt.legend(fontsize=10, loc="upper right", title="Policy")

            # Format
            plt.title(metric, fontsize=15, pad=25)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().grid(axis="y", color="lightgrey")
            plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                                  width=1)
            plt.gca().spines["bottom"].set_linewidth(2)
            plt.gca().spines["bottom"].set_color("lightgrey")
            # plt.gca().set_ylim(y_limits[metric])
            plt.gca().set_xlim(x_limits[metric])
            plt.gca().set_xticks(list(valid_t.keys()))
            plt.xlabel("Threshold (t)", labelpad=8)
            plt.ylabel(metric, labelpad=8)

            plt.tight_layout()
            plt.savefig(f"")

            plt.close()

    def randaug_balance():
        valid_M = {1: 0, 5: 1, 9: 2, 13: 3}
        valid_N = {1: 0, 2: 1, 3: 2, 4: 3}
        valid_t = {100: 0, 200: 1, 300: 2, 400: 3, 600: 4, 800: 5, 1000: 6}

        """
        search_string = "randaug_balance"
        aggregated_metrics = pd.DataFrame(columns=["t", "N", "M", *columns])

        # Get files
        for experiment in os.listdir(source):
            if search_string in experiment:
                # Get N, M, and t
                N = int(experiment[experiment.find("N") + 1])
                M = int(experiment[experiment.find("M") + 1:len(experiment)])
                t = int(experiment[16:experiment.find("_", 16, len(experiment))])

                if N not in valid_N.keys() or M not in valid_M.keys() or t not in valid_t.keys():
                    continue

                metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
                averaged_metrics = average_metrics(metrics)
                new_row = {"t": t, "N": N, "M": M}
                for column in columns:
                    new_row[column] = round(averaged_metrics[column].mean(), 4)
                aggregated_metrics = aggregated_metrics.append(new_row, ignore_index=True)

        aggregated_metrics = aggregated_metrics.rename(
            columns={"t": "Threshold", "N": "Sequence", "M": "Magnitude", "val_accuracy": "Accuracy",
                     "val_covid_recall": "Covid Recall", "val_lung_opacity_recall": "Lung Opacity Recall",
                     "val_normal_recall": "Normal Recall", "val_viral_pneumonia_recall": "Viral Pneumonia Recall", })
        aggregated_metrics["Threshold"] = aggregated_metrics["Threshold"].astype(int)
        aggregated_metrics["Sequence"] = aggregated_metrics["Sequence"].astype(int)
        aggregated_metrics["Magnitude"] = aggregated_metrics["Magnitude"].astype(int)
        aggregated_metrics.to_csv(
            "", index=False)
        """

        aggregated_metrics = pd.read_csv(
            "")

        best_aggregated_metrics = pandas.DataFrame(columns=aggregated_metrics.columns)
        for n in valid_N.keys():
            for threshold in valid_t.keys():
                threshold_metrics = aggregated_metrics[
                    (aggregated_metrics.Sequence == n) & (aggregated_metrics.Threshold == threshold)]
                best_aggregated_metrics = best_aggregated_metrics.append(
                    aggregated_metrics.loc[[threshold_metrics["Accuracy"].idxmax()]])

        metric_map = {"val_accuracy": "Accuracy", "val_covid_recall": "Covid Recall",
                      "val_lung_opacity_recall": "Lung Opacity Recall", "val_normal_recall": "Normal Recall",
                      "val_viral_pneumonia_recall": "Viral Pneumonia Recall"}
        metric_map = {v: k for k, v in metric_map.items()}
        baselines = {"Accuracy": 0.6979, "Covid Recall": 0.4222, "Lung Opacity Recall": 0.6200, "Normal Recall": 0.8704,
                     "Viral Pneumonia Recall": 0.4758}
        y_limits = {"Accuracy": (0, 1), "Covid Recall": (0, 1), "Lung Opacity Recall": (0, 0),
                    "Normal Recall": (0, 1), "Viral Pneumonia Recall": (0, 1)}
        x_limits = {"Accuracy": (100, 1000), "Covid Recall": (100, 1000), "Lung Opacity Recall": (100, 1000),
                    "Normal Recall": (100, 1000), "Viral Pneumonia Recall": (100, 1000)}
        for metric in best_aggregated_metrics.columns[-5:]:
            # Lines
            sns.lineplot(data=best_aggregated_metrics,
                         x="Threshold",
                         y=metric,
                         hue="Sequence",
                         palette=sns.color_palette("hls", len(valid_N.keys())),
                         linewidth=1,
                         alpha=0.8,
                         )

            sns.regplot(data=best_aggregated_metrics,
                        x="Threshold",
                        y=metric,
                        color="red",
                        scatter=False,
                        ci=None,
                        line_kws={"linestyle": "dashed", "linewidth": 2},
                        lowess=True,
                        )

            # Baseline
            plt.plot(valid_t.keys(),
                     [baselines[metric]] * len(valid_t.keys()),
                     color="green",
                     linewidth=2,
                     linestyle="dashed")

            # Legend
            plt.legend(fontsize=10, loc="upper right", title="Sequence")

            # Format
            plt.title(metric, fontsize=15, pad=25)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().grid(axis="y", color="lightgrey")
            plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                                  width=1)
            plt.gca().spines["bottom"].set_linewidth(2)
            plt.gca().spines["bottom"].set_color("lightgrey")
            # plt.gca().set_ylim(y_limits[metric])
            plt.gca().set_xlim(x_limits[metric])
            plt.gca().set_xticks(list(valid_t.keys()))
            plt.xlabel("Threshold (t)", labelpad=8)
            plt.ylabel(metric, labelpad=8)

            plt.tight_layout()
            plt.savefig(f"")
            plt.close()

    def ganaug_balance():
        valid_arch = {"ACGAN": 0, "ADCGAN": 1, "BigGAN": 2, "ContraGAN": 3, "cStyleGAN2": 4, "MHGAN": 5, "ReACGAN": 6}
        valid_t = {100: 0, 200: 1, 300: 2, 400: 3, 600: 4, 800: 5, 1000: 6}

        """search_string = "ganaug_with_normalization_balance_"
        aggregated_metrics = pd.DataFrame(columns=["t", "Architecture", *columns])

        # Get files
        for experiment in os.listdir(source):
            if search_string in experiment:
                # Get t and a
                t = int(experiment[len(search_string): experiment.find("_", len(search_string), len(experiment))])
                for arq in valid_arch:
                    if arq in experiment:
                        if arq == "ACGAN" and "ReACGAN" in experiment:
                            continue
                        a = arq
                        break

                if a not in valid_arch.keys() or t not in valid_t.keys():
                    continue

                metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
                averaged_metrics = average_metrics(metrics)
                new_row = {"t": t, "Architecture": a}
                for column in columns:
                    new_row[column] = round(averaged_metrics[column].mean(), 4)
                aggregated_metrics = aggregated_metrics.append(new_row, ignore_index=True)

        aggregated_metrics = aggregated_metrics.rename(
            columns={"t": "Threshold", "Architecture": "Architecture", "val_accuracy": "Accuracy",
                     "val_covid_recall": "Covid Recall", "val_lung_opacity_recall": "Lung Opacity Recall",
                     "val_normal_recall": "Normal Recall", "val_viral_pneumonia_recall": "Viral Pneumonia Recall", })
        aggregated_metrics.to_csv(
            "",
            index=False)"""

        aggregated_metrics = pd.read_csv(
            "")

        metric_map = {"val_accuracy": "Accuracy", "val_covid_recall": "Covid Recall",
                      "val_lung_opacity_recall": "Lung Opacity Recall", "val_normal_recall": "Normal Recall",
                      "val_viral_pneumonia_recall": "Viral Pneumonia Recall"}
        metric_map = {v: k for k, v in metric_map.items()}
        baselines = {"Accuracy": 0.6979, "Covid Recall": 0.4222, "Lung Opacity Recall": 0.6200, "Normal Recall": 0.8704,
                     "Viral Pneumonia Recall": 0.4758}
        y_limits = {"Accuracy": (0.68, 0.8), "Covid Recall": (0.4, 0.7), "Lung Opacity Recall": (0.6, 0.8),
                    "Normal Recall": (0.70, 0.90), "Viral Pneumonia Recall": (0.4, 0.9)}
        x_limits = {"Accuracy": (100, 1000), "Covid Recall": (100, 1000), "Lung Opacity Recall": (100, 1000),
                    "Normal Recall": (100, 1000), "Viral Pneumonia Recall": (100, 1000)}
        for metric in aggregated_metrics.columns[-5:]:
            # Lines
            sns.lineplot(data=aggregated_metrics,
                         x="Threshold",
                         y=metric,
                         hue="Architecture",
                         palette=sns.color_palette("hls", len(valid_arch.keys())),
                         legend="full",
                         linewidth=1,
                         alpha=0.8
                         )

            sns.regplot(data=aggregated_metrics,
                        x="Threshold",
                        y=metric,
                        color="red",
                        scatter=False,
                        ci=None,
                        line_kws={"linestyle": "dashed", "linewidth": 2},
                        lowess=True,
                        )

            # Baseline
            plt.plot(valid_t.keys(),
                     [baselines[metric]] * len(valid_t.keys()),
                     color="green",
                     linewidth=2,
                     linestyle="dashed")

            # Legend
            plt.legend(fontsize=10, loc="upper right", title="Architecture")

            # Format
            plt.title(metric, fontsize=15, pad=25)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().grid(axis="y", color="lightgrey")
            plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                                  width=1)
            plt.gca().spines["bottom"].set_linewidth(2)
            plt.gca().spines["bottom"].set_color("lightgrey")
            # plt.gca().set_ylim(y_limits[metric])
            plt.gca().set_xlim(x_limits[metric])
            plt.gca().set_xticks(list(valid_t.keys()))
            plt.xlabel("Threshold (t)", labelpad=8)
            plt.ylabel(metric, labelpad=8)

            plt.tight_layout()
            plt.savefig(
                f"")

            plt.close()

    def transform():
        valid_M = {1: 0, 7: 1, 13: 2}
        valid_transforms = {"shear_x": 10, "shear_y": 11, "translate_x": 13, "translate_y": 14, "rotate": 8,
                            "brightness": 1, "contrast": 2, "sharpness": 9, "posterize": 7, "solarize": 12,
                            "auto_contrast": 0, "equalize": 3, "perspective": 6, "horizontal_flip": 5,
                            "vertical_flip": 15,
                            "gaussian_blur": 4}

        readable_map = {"shear_x": "Horizontal Shear", "shear_y": "Vertical Shear",
                        "translate_x": "Horizontal Translation",
                        "translate_y": "Vertical Translation", "rotate": "Rotate", "brightness": "Brightness",
                        "contrast": "Contrast", "sharpness": "Sharpness", "posterize": "Posterize",
                        "solarize": "Solarize",
                        "auto_contrast": "Auto Contrast", "equalize": "Equalize", "perspective": "Perspective",
                        "horizontal_flip": "Horizontal Flip", "vertical_flip": "Vertical Flip",
                        "gaussian_blur": "Gaussian Blur"}

        """
        aggregated_metrics = pd.DataFrame(columns=["transform", "M", *columns])
        # Get files
        for experiment in os.listdir(source):
            for transform in valid_transforms.keys():
                if not experiment.startswith(transform) or "balance" in experiment:
                    continue

                # Get M
                M = int(experiment[experiment.find("M") + 1:len(experiment)])

                if M not in valid_M.keys():
                    continue

                metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
                averaged_metrics = average_metrics(metrics)
                new_row = {"transform": readable_map[transform], "M": M}
                for column in columns:
                    new_row[column] = round(averaged_metrics[column].mean(), 4)
                aggregated_metrics = aggregated_metrics.append(new_row, ignore_index=True)

        aggregated_metrics = aggregated_metrics.rename(
            columns={"transform": "Transformation", "M": "Magnitude", "val_accuracy": "Accuracy",
                     "val_covid_recall": "Covid Recall", "val_lung_opacity_recall": "Lung Opacity Recall",
                     "val_normal_recall": "Normal Recall", "val_viral_pneumonia_recall": "Viral Pneumonia Recall", })
        aggregated_metrics.to_csv(
            "", index=False)
        """

        aggregated_metrics = pd.read_csv(
            "")

        for transform in valid_transforms.keys():
            filtered_aggregated_metrics = aggregated_metrics[
                aggregated_metrics.Transformation == readable_map[transform]]

            # Lines
            sns.lineplot(data=filtered_aggregated_metrics,
                         x="Magnitude",
                         y="Accuracy",
                         palette=sns.color_palette("hls", 5)[0],
                         legend="full",
                         linewidth=1,
                         )

            # Baseline
            plt.plot(valid_M.keys(), [0.8769] * len(valid_M.keys()), color="green", linewidth=2,
                     linestyle="dashed")

            # Format
            plt.title(f"{readable_map[transform]}", fontsize=15, pad=25)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().grid(axis="y", color="lightgrey")
            plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                                  width=1)
            plt.gca().spines["bottom"].set_linewidth(2)
            plt.gca().spines["bottom"].set_color("lightgrey")
            plt.gca().set_ylim((0.85, 0.9))
            plt.gca().set_xlim((1, 13))
            # plt.gca().set_yticks()
            plt.gca().set_xticks(list(valid_M.keys()))
            plt.xlabel("Magnitude (M)", labelpad=8)
            plt.ylabel("Accuracy", labelpad=8)

            plt.tight_layout()
            plt.savefig(
                f"")
            plt.close()

        """rearranged_aggregated_metrics = pd.DataFrame(columns=[*aggregated_metrics.columns[:-5], "Metric", "Value"])
        for idx, row in aggregated_metrics.iterrows():
            for metric in aggregated_metrics.columns[-5:]:
                new_row = {"Transformation": row["Transformation"], "Magnitude": row["Magnitude"], "Metric": metric,
                           "Value": row[metric]}
                rearranged_aggregated_metrics = rearranged_aggregated_metrics.append(new_row, ignore_index=True)

        for transform in valid_transforms.keys():
            filtered_aggregated_metrics = rearranged_aggregated_metrics[
                rearranged_aggregated_metrics.Transformation == readable_map[transform]]

            # Lines
            sns.lineplot(data=filtered_aggregated_metrics,
                         x="Magnitude",
                         y="Value",
                         hue="Metric",
                         palette=sns.color_palette("hls", 5),
                         legend="full",
                         linewidth=1,
                         )

            # Baseline
            plt.plot(valid_M.keys(), [0.8769] * len(valid_M.keys()), color=sns.color_palette("hls", 5)[0], linewidth=2,
                     linestyle="dashed")
            plt.plot(valid_M.keys(), [0.8278] * len(valid_M.keys()), color=sns.color_palette("hls", 5)[1], linewidth=2,
                     linestyle="dashed")
            plt.plot(valid_M.keys(), [0.8333] * len(valid_M.keys()), color=sns.color_palette("hls", 5)[2], linewidth=2,
                     linestyle="dashed")
            plt.plot(valid_M.keys(), [0.9155] * len(valid_M.keys()), color=sns.color_palette("hls", 5)[3], linewidth=2,
                     linestyle="dashed")
            plt.plot(valid_M.keys(), [0.9110] * len(valid_M.keys()), color=sns.color_palette("hls", 5)[4], linewidth=2,
                     linestyle="dashed")

            # Legend
            plt.legend(fontsize=10, loc="upper right", title="Metric")

            # Format
            plt.title(f"{readable_map[transform]}", fontsize=15, pad=25)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().grid(axis="y", color="lightgrey")
            plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                                  width=1)
            plt.gca().spines["bottom"].set_linewidth(2)
            plt.gca().spines["bottom"].set_color("lightgrey")
            # plt.gca().set_ylim((0.74, 0.96))
            plt.gca().set_xlim((1, 13))
            # plt.gca().set_yticks()
            plt.gca().set_xticks(list(valid_M.keys()))
            plt.xlabel("Magnitude (M)", labelpad=8)
            plt.ylabel("Value", labelpad=8)

            plt.tight_layout()
            plt.savefig(
                f"")
            plt.close()"""

    def autoaug():
        pass

    def randaug():
        valid_M = {1: 0, 5: 1, 9: 2, 13: 3}
        valid_N = {1: 0, 2: 1, 3: 2, 4: 3}

        search_string = "randaug"
        aggregated_metrics = pd.DataFrame(columns=["N", "M", *columns])

        # Get files
        for experiment in os.listdir(source):
            if search_string in experiment and "balance" not in experiment and "customrandaug" not in experiment:
                # Get N, M
                N = int(experiment[experiment.find("N") + 1])
                M = int(experiment[experiment.find("M") + 1:len(experiment)])

                if N not in valid_N.keys() or M not in valid_M.keys():
                    print(f"skipping N{N}M{M}")
                    continue

                metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
                averaged_metrics = average_metrics(metrics)
                new_row = {"N": N, "M": M}
                for column in columns:
                    new_row[column] = round(averaged_metrics[column].mean(), 4)
                aggregated_metrics = aggregated_metrics.append(new_row, ignore_index=True)

        aggregated_metrics = aggregated_metrics.rename(
            columns={"N": "Sequence", "M": "Magnitude", "val_accuracy": "Accuracy",
                     "val_covid_recall": "Covid Recall", "val_lung_opacity_recall": "Lung Opacity Recall",
                     "val_normal_recall": "Normal Recall", "val_viral_pneumonia_recall": "Viral Pneumonia Recall"})

        aggregated_metrics["Sequence"] = aggregated_metrics["Sequence"].astype(int)
        aggregated_metrics["Magnitude"] = aggregated_metrics["Magnitude"].astype(int)

        aggregated_metrics.to_csv(
            "", index=False)

        aggregated_metrics = pd.read_csv(
            "")

        metric_map = {"val_accuracy": "Accuracy", "val_covid_recall": "Covid Recall",
                      "val_lung_opacity_recall": "Lung Opacity Recall", "val_normal_recall": "Normal Recall",
                      "val_viral_pneumonia_recall": "Viral Pneumonia Recall"}
        metric_map = {v: k for k, v in metric_map.items()}
        baselines = {"Accuracy": 0.8769, "Covid Recall": 0.8278, "Lung Opacity Recall": 0.8333, "Normal Recall": 0.9155,
                     "Viral Pneumonia Recall": 0.9110}
        y_limits = {"Accuracy": (0, 1), "Covid Recall": (0, 1), "Lung Opacity Recall": (0, 0),
                    "Normal Recall": (0, 1), "Viral Pneumonia Recall": (0, 1)}
        x_limits = {"Accuracy": (1, 13), "Covid Recall": (1, 13), "Lung Opacity Recall": (1, 13),
                    "Normal Recall": (1, 13), "Viral Pneumonia Recall": (1, 13)}
        for metric in aggregated_metrics.columns[-5:]:
            # Lines
            sns.lineplot(data=aggregated_metrics,
                         x="Magnitude",
                         y=metric,
                         hue="Sequence",
                         palette=sns.color_palette("hls", len(valid_N.keys())),
                         linewidth=1,
                         )

            # Baseline
            plt.plot(valid_M.keys(),
                     [baselines[metric]] * len(valid_M.keys()),
                     color="green",
                     linewidth=2,
                     linestyle="dashed")

            # Legend
            plt.legend(fontsize=10, loc="upper right", title="Sequence")

            # Format
            plt.title(metric, fontsize=15, pad=25)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().grid(axis="y", color="lightgrey")
            plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                                  width=1)
            plt.gca().spines["bottom"].set_linewidth(2)
            plt.gca().spines["bottom"].set_color("lightgrey")
            # plt.gca().set_ylim(y_limits[metric])
            plt.gca().set_xlim(x_limits[metric])
            plt.gca().set_xticks(list(valid_M.keys()))
            plt.xlabel("Magnitude (N)", labelpad=8)
            plt.ylabel(metric, labelpad=8)

            plt.tight_layout()
            plt.savefig(f"")
            plt.close()

    def ganaug():
        valid_arch = {"ACGAN": 0, "ADCGAN": 1, "BigGAN": 2, "ContraGAN": 3, "cStyleGAN2": 4, "MHGAN": 5, "ReACGAN": 6}
        valid_p = {0.1: 0, 0.3: 1, 0.5: 2, 0.7: 3, 0.9: 4}

        """search_string = "ganaug_with_normalization"
        aggregated_metrics = pd.DataFrame(columns=["Probability", "Architecture", *columns])

        # Get files
        for experiment in os.listdir(source):
            if search_string in experiment and "balance" not in experiment:
                # Get t and a
                dot = experiment.find(".")
                p = float(experiment[dot - 1: dot + 2])
                for arch in valid_arch:
                    if arch in experiment:
                        if arch == "ACGAN" and "ReACGAN" in experiment:
                            continue
                        break

                if arch not in valid_arch.keys() or p not in valid_p.keys():
                    continue

                metrics = pd.read_csv(os.path.join(os.path.join(source, experiment), experiment + ".csv"))
                averaged_metrics = average_metrics(metrics)
                new_row = {"Probability": p, "Architecture": arch}
                for column in columns:
                    new_row[column] = round(averaged_metrics[column].mean(), 4)
                aggregated_metrics = aggregated_metrics.append(new_row, ignore_index=True)

        aggregated_metrics = aggregated_metrics.rename(
            columns={"Probability": "Probability", "Architecture": "Architecture", "val_accuracy": "Accuracy",
                     "val_covid_recall": "Covid Recall", "val_lung_opacity_recall": "Lung Opacity Recall",
                     "val_normal_recall": "Normal Recall", "val_viral_pneumonia_recall": "Viral Pneumonia Recall", })
        aggregated_metrics.to_csv(
            "",
            index=False)"""

        aggregated_metrics = pd.read_csv(
            "D:/Uni/Thesis/Code/Classifiers/saves/images/graphs/ganaug/aggregated_metrics.csv")

        baselines = {"Accuracy": 0.8769, "Covid Recall": 0.8278, "Lung Opacity Recall": 0.8333, "Normal Recall": 0.9155,
                     "Viral Pneumonia Recall": 0.9110}
        for arch in ["ACGAN", "ContraGAN", "cStyleGAN2", "ADCGAN", "BigGAN", "MHGAN", "ReACGAN"]:
            if arch in ["ADCGAN", "BigGAN", "MHGAN", "ReACGAN"]:
                y_limits = {"Accuracy": (0.85, 0.9), "Covid Recall": (0.8, 0.87), "Lung Opacity Recall": (0.82, 0.88),
                            "Normal Recall": (0.89, 0.94), "Viral Pneumonia Recall": (0.83, 0.95)}
            else:
                y_limits = {"Accuracy": (0.75, 0.89), "Covid Recall": (0.50, 0.86), "Lung Opacity Recall": (0.75, 0.86),
                            "Normal Recall": (0.86, 0.94), "Viral Pneumonia Recall": (0.65, 0.94)}
            for metric in aggregated_metrics.columns[-5:]:
                filtered_aggregated_metrics = aggregated_metrics[aggregated_metrics.Architecture == arch]

                sns.lineplot(data=filtered_aggregated_metrics,
                             x="Probability",
                             y=metric,
                             palette=sns.color_palette("hls", 1),
                             legend="full",
                             linewidth=2,
                             )

                # Baseline
                plt.plot(valid_p.keys(),
                         [baselines[metric]] * len(valid_p.keys()),
                         color="green",
                         linewidth=2,
                         linestyle="dashed")

                # Format
                # plt.title(f"{arch} - {metric}", fontsize=15, pad=25)
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().grid(axis="y", color="lightgrey")
                plt.gca().tick_params(color="lightgrey", labelcolor="black", left=False, direction="inout", length=10,
                                      width=1)
                plt.gca().spines["bottom"].set_linewidth(2)
                plt.gca().spines["bottom"].set_color("lightgrey")
                plt.gca().set_ylim(y_limits[metric])
                plt.gca().set_xlim((0.1, 0.9))
                plt.gca().set_xticks(list(valid_p.keys()))
                plt.xlabel("Probability (p)", labelpad=8)
                plt.ylabel(metric, labelpad=8)

                plt.tight_layout()
                plt.savefig(f"")

                plt.close()

    source = ""
    columns = ["val_accuracy", "val_covid_recall", "val_lung_opacity_recall", "val_normal_recall",
               "val_viral_pneumonia_recall"]
    num_splits = 5
    num_epochs = 70

    ganaug()


classifiers()
