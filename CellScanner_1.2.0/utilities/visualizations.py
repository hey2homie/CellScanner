import math
import os
from itertools import cycle
from datetime import datetime

import numpy as np
import umap
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, \
    PrecisionRecallDisplay, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


class UmapVisualization:
    """
    UmapVisualization class is used to create UMAP embeddings of the dataframe that are used in low-dimensional
    projection of the results as well as the part of the training dataset.
    """
    def __init__(self, data: np.ndarray, num_cores: int, dims: int) -> None:
        """
        Args:
            data (np.array): dataframe.
            num_cores: amount of CPU cores used for the computation of embeddings
        """
        self.data = data
        self.num_cores = num_cores
        self.dims = dims
        self.embeddings = self.__reduction()

    def __reduction(self) -> np.ndarray:
        """
        Performs dimensional reduction to either 2D or 3D as specified in the settings.
        Returns:
            fitted (np.array): calculated embeddings.
        """
        reducer = umap.UMAP(n_components=self.dims, n_neighbors=25, min_dist=0.1, metric="euclidean",
                            n_jobs=self.num_cores)
        fitted = reducer.fit_transform(self.data)
        return fitted

    def get_embeddings(self) -> np.ndarray:
        """
        Returns:
            embeddings (np.array): calculated embeddings.
        """
        return self.embeddings


class MplVisualization:

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        datestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.output_path = self.output_path + "/" + datestamp + "/"
        self.colors = cycle(["aqua", "darkorange", "cornflowerblue", "goldenrod", "rosybrown", "lightgreen",
                             "lightgray", "orchid", "darkmagenta", "olive"])    # 10 colors should be enough
        os.mkdir(self.output_path)

    def diagnostics(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
        self.__roc(true_labels, predicted_labels)
        self.__precision_recall(true_labels, predicted_labels)
        self.__confusion_matrix(true_labels, predicted_labels)

    def __roc(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = true_labels.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predicted_labels[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.figure()
        colors = self.colors
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))
        # TODO: Change title from "Class N" to name of bacteria (above)
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.savefig(self.output_path + "/roc.png")

    def __precision_recall(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
        n_classes = true_labels.shape[1]
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(true_labels[:, i], predicted_labels[:, i])
            average_precision[i] = average_precision_score(true_labels[:, i], predicted_labels[:, i])
        colors = self.colors
        _, ax = plt.subplots(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
        for i, color in zip(range(n_classes), colors):
            display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i],
                                             average_precision=average_precision[i])
            display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)
            # TODO: Change title from "Class N" to name of bacteria (above)
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["Iso-f1 curves"])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Precision-Recall Curves")
        plt.savefig(self.output_path + "precision_recall.png")

    def __confusion_matrix(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> None:
        n_classes = true_labels.shape[1]
        rows = int(np.floor(np.sqrt(n_classes)))
        cols = int(np.ceil(n_classes / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.ravel()
        for i in range(n_classes):
            disp = ConfusionMatrixDisplay(confusion_matrix(true_labels[:, i], predicted_labels[:, i]),
                                          display_labels=[0, i])
            disp.plot(ax=axes[i], values_format='.4g')
            disp.ax_.set_title(f'class {i}')    # TODO: Change title from "Class N" to name of bacteria
            if i < (rows - 1) * cols:
                disp.ax_.set_xlabel('')
            if i % cols != 0:
                disp.ax_.set_ylabel('')
            disp.im_.colorbar.remove()
            plt.subplots_adjust(wspace=0.10, hspace=0.1)
            fig.colorbar(disp.im_, ax=axes)
        # TODO: Remove last plot if rows * cols > n_classes
        # TODO: Heat-bar sometimes looks weird
        plt.title("Confusion Matrices")
        plt.savefig(self.output_path + "confusion_matrix.png")


class TrainingVisualization:
    # Ideas: custom callback to draw matplotlib plot or plotly (preferably plotly) after each epoch. Otherwise, consider
    # using a tensorboard callback.
    pass
