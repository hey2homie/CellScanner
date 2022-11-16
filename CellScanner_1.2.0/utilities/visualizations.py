import os
from itertools import cycle
from datetime import datetime

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, \
    PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from mlxtend.evaluate import confusion_matrix

import umap
import matplotlib as mpl
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

from utilities.settings import Settings, SettingsOptions, ModelsInfo


class UmapVisualization:

    def __init__(self, data: np.ndarray, num_cores: int, dims: int) -> None:
        self.data = data
        self.num_cores = num_cores
        self.dims = dims
        self.embeddings = self.__reduction()

    def __reduction(self) -> np.ndarray:
        reducer = umap.UMAP(n_components=self.dims, n_neighbors=25, min_dist=0.1, metric="euclidean",
                            n_jobs=self.num_cores)
        fitted = reducer.fit_transform(self.data)
        return fitted

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings


class MplVisualization:

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path + "/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "/"
        os.mkdir(self.output_path)
        # 10 colors should be enough, I guess
        self.colors = cycle(["aqua", "darkorange", "cornflowerblue", "goldenrod", "rosybrown", "lightgreen",
                             "lightgray", "orchid", "darkmagenta", "olive"])
        self.classes = None
        self.n_classes = None

    def save_predictions_visualizations(self, inputs: dict, settings: Settings) -> None:
        if settings.vis_type == "Channels":
            if settings.fc_type == "Accuri":
                channels_use = settings.vis_channels_accuri
                indexes = [SettingsOptions.vis_channels_accuri.value.index(channel) for channel in channels_use]
            else:
                channels_use = settings.vis_channels_cytoflex
                indexes = [SettingsOptions.vis_channels_cytoflex.value.index(channel) for channel in channels_use]
        else:
            indexes = [-3, -2, -1]
            channels_use = ["X", "Y", "Z"]

        for file, data in inputs.items():
            data, mse = data[0], data[1]
            if settings.vis_type == "Channels":
                labels = np.unique(data[:, -2])
            else:
                labels = np.unique(data[:, -5])
            file = file.split("/")[-1].split(".")[0]
            fig = plt.figure(figsize=(7, 7))
            cmap = plt.cm.jet
            if settings.vis_dims == "3":
                ax = fig.add_subplot(projection="3d")
                bounds = np.linspace(0, len(labels), len(labels) + 1)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            else:
                ax = fig.add_subplot()
            mse_fig = plt.figure(figsize=(7, 7))
            ax_mse = mse_fig.add_subplot()
            col_x = list(map(float, data[:, indexes[0]]))
            col_y = list(map(float, data[:, indexes[1]]))
            mse = list(mse)
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)
            colors = cmap(np.linspace(0, 1, len(labels)))
            for _, (label, color) in enumerate(zip(labels, colors), 1):
                if settings.vis_type == "Channels":
                    indexes_plot = np.where(data[:, -2] == label)
                else:
                    indexes_plot = np.where(data[:, -5] == label)
                axis_x = np.take(col_x, indexes_plot, axis=0)
                axis_y = np.take(col_y, indexes_plot, axis=0)
                if settings.vis_dims == "3":
                    col_z = list(map(float, data[:, indexes[2]]))
                    axis_z = np.take(col_z, indexes_plot, axis=0)
                    ax.scatter3D(axis_x, axis_y, axis_z, c=color, label=label, norm=norm)
                else:
                    ax.scatter(axis_x, axis_y, c=color, label=label)
                mse_axis_y = np.take(mse, indexes_plot, axis=0)
                ax_mse.scatter(indexes_plot, mse_axis_y, c=color, label=label)
            ax.set_xlabel(channels_use[0])
            ax.set_ylabel(channels_use[1])
            ax_mse.set_xlabel("Index")
            ax_mse.set_ylabel("MSE")
            ax_mse.axhline(y=settings.mse_threshold, color="r", linestyle="-")
            ax_mse.set_title("Reconstruction Error for " + file)
            if settings.vis_dims == "3":
                ax.set_zlabel(channels_use[2])
                ax.set_title("3D Scatter Plot of " + file)
            else:
                ax.set_title("2D Scatter Plot of " + file)
            ax.legend()
            ax_mse.legend()
            fig.savefig(self.output_path + file + "_" + "predictions.png")
            mse_fig.savefig(self.output_path + file + "_" + "mse.png")
            plt.close()

    def diagnostics(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> list:
        if np.unique(true_labels).shape[0] != np.unique(predicted_labels).shape[0]:
            raise ValueError("Number of classes in true and predicted labels are not equal")
        self.classes = np.unique(true_labels)
        self.n_classes = self.classes.shape[0]
        true_labels_binarized = label_binarize(true_labels, classes=self.classes)
        predicted_labels_binarized = label_binarize(predicted_labels, classes=self.classes)
        self.classes = {i: bacteria_class for i, bacteria_class in enumerate(self.classes)}
        labels_compared = self.__pie(true_labels, predicted_labels)
        self.__roc(true_labels_binarized, predicted_labels_binarized)
        self.__precision_recall(true_labels_binarized, predicted_labels_binarized)
        self.__confusion_matrices(true_labels_binarized, predicted_labels_binarized)
        self.__aggregated_matrix(true_labels, predicted_labels)
        return labels_compared

    def __pie(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> list:
        labels = []
        for i in range(0, len(true_labels)):
            labels.append("Correct") if true_labels[i] == predicted_labels[i] else labels.append("Incorrect")
        _, count_labels = np.unique(labels, return_counts=True)
        plt.pie(count_labels, labels=["Correct", "Incorrect"])
        plt.title("Pie Chart of Prediction Accuracy")
        plt.savefig(self.output_path + "pie.png")
        return labels

    def __roc(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predicted_labels[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.figure()
        for i, color in zip(range(self.n_classes), self.colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label="ROC curve of class {0} (area = {1:0.2f})".format(self.classes[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.savefig(self.output_path + "roc.png")

    def __precision_recall(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(self.n_classes):
            precision[i], recall[i], _ = precision_recall_curve(true_labels[:, i], predicted_labels[:, i])
            average_precision[i] = average_precision_score(true_labels[:, i], predicted_labels[:, i])
        _, ax = plt.subplots(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        l = None
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
        display = None
        for i, color in zip(range(self.n_classes), self.colors):
            display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i],
                                             average_precision=average_precision[i])
            display.plot(ax=ax, name=f"Precision-recall for class {self.classes[i]}", color=color)
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["Iso-f1 curves"])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Precision-Recall Curves")
        plt.savefig(self.output_path + "precision_recall.png")

    def __confusion_matrices(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
        rows = int(np.floor(np.sqrt(self.n_classes)))
        cols = int(np.ceil(self.n_classes / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        try:
            axes = axes.ravel()
        except AttributeError:
            pass
        for i in range(self.n_classes):
            disp = ConfusionMatrixDisplay(confusion_matrix(true_labels[:, i], predicted_labels[:, i]),
                                          display_labels=[0, i])
            disp.plot(ax=axes[i], values_format=".4g", colorbar=False)
            disp.ax_.set_title(f"{self.classes[i]}")
            if i < (rows - 1) * cols:
                disp.ax_.set_xlabel("")
            if i % cols != 0:
                disp.ax_.set_ylabel("")
            plt.subplots_adjust(wspace=0.10, hspace=0.1)
        # TODO: Remove last plot if rows * cols > n_classes
        plt.savefig(self.output_path + "confusion_matrices.png")

    def __aggregated_matrix(self, true_labels: np.ndarray, predicted_labels: np.ndarray):
        _, true_labels = np.unique(true_labels, return_inverse=True)
        _, predicted_labels = np.unique(predicted_labels, return_inverse=True)
        cm = confusion_matrix(y_target=true_labels, y_predicted=predicted_labels, binary=False)
        _, _ = plot_confusion_matrix(conf_mat=cm, class_names=list(self.classes.values()), figsize=(7, 8))
        plt.title("Confusion Matrix")
        plt.savefig(self.output_path + "confusion_matrix.png")
