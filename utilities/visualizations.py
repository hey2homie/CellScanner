from itertools import cycle

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize
from mlxtend.evaluate import confusion_matrix

import umap
import matplotlib as mpl
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

from .helpers import drop_blanks
from .settings import Settings, SettingsOptions


class UmapVisualization:
    """
    UmapVisualization class is used for the dimensionality reduction of the data using UMAP algorithm. Embeddings are
    used for the visualization of the data in the 2D or 3D space.

    Attributes:
    ----------
    data: np.ndarray
        Data to be reduced.
    num_cores: int
        Number of CPU cores to be used for the reduction.
    dims: int
        Number of dimensions to be reduced to.
    """

    def __init__(self, data: np.ndarray, num_cores: int, dims: int) -> None:
        """
        Args:
            data (np.ndarray): Data to be reduced.
            num_cores (int): Number of CPU cores to be used for the reduction.
            dims (int): Number of dimensions to be reduced to.
        Returns:
            None.
        """
        self.data = data
        self.num_cores = num_cores
        self.dims = dims
        self.embeddings = self.__reduction()

    def __reduction(self) -> np.ndarray:
        """
        Performs the dimensionality reduction and returns the embeddings.
        Returns:
            np.ndarray: Embeddings of the data.
        """
        reducer = umap.UMAP(
            n_components=self.dims,
            n_neighbors=25,
            min_dist=0.1,
            metric="euclidean",
            n_jobs=self.num_cores,
        )
        fitted = reducer.fit_transform(self.data)
        return fitted


class MplVisualization:
    """
    MplVisualization class is used for the visualization of the data using matplotlib library. Visualizations are done
    for both prediction results and for the model diagnostics. First includes scatter plot of the embeddings or
    channels, as well as the scatter plot of the reconstruction error if the autoencoder is used. For the diagnostics,
    ROC curve, precision-recall curve, confusion matrix, aggregated confusion matrix, and pie chart of overall accuracy
    are plotted.

    Attributes:
    ----------
    output_path: str
        Path to the output directory.
    colors: cycle
        Colors for the plots. Assumes that no more than 10 classes are present in the data.
    classes: np.ndarray
        Array of classes.
    n_classes: int
        Number of classes.
    """

    def __init__(self, output_path: str) -> None:
        """
        Assumes that no more than 10 classes are present in the data.
        Args:
            output_path (str): Path to the output directory.
        Returns:
            None.
        """
        self.output_path = output_path
        self.colors = cycle(
            [
                "aqua",
                "darkorange",
                "cornflowerblue",
                "goldenrod",
                "rosybrown",
                "lightgreen",
                "lightgray",
                "orchid",
                "darkmagenta",
                "olive",
            ]
        )
        self.classes = None
        self.n_classes = None

    def save_predictions_visualizations(self, inputs: dict, settings: Settings) -> None:
        """
        Saves the visualizations of the prediction results. Can be either 2D or 3D scatter plots.
        Args:
            inputs (dict): Dictionary containing the results of prediction.
            settings (Settings): Settings object containing the settings of the model.
        Returns:
            None.
        """
        for file, data in inputs.items():
            dataframe, gating_results, labels = (
                data["data"],
                data["gating_results"],
                data["labels"],
            )
            labels_uniq = np.unique(labels)
            file = file.split("/")[-1].split(".")[0]
            if settings.vis_type == "Channels":
                if settings.fc_type == "Accuri":
                    channels_use = settings.vis_channels_accuri
                    indexes = [
                        SettingsOptions.vis_channels_accuri.value.index(channel)
                        for channel in channels_use
                    ]
                else:
                    channels_use = settings.vis_channels_cytoflex
                    indexes = [
                        SettingsOptions.vis_channels_cytoflex.value.index(channel)
                        for channel in channels_use
                    ]
            else:
                dataframe = data["embeddings"]
                channels_use = ["X", "Y", "Z"]
                indexes = [0, 1, 2]
            fig = plt.figure(figsize=(7, 7))
            cmap = plt.cm.jet
            if settings.vis_dims == 3:
                ax = fig.add_subplot(projection="3d")
                bounds = np.linspace(0, len(labels_uniq), len(labels_uniq) + 1)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            else:
                ax = fig.add_subplot()
            if settings.gating_type == "Autoencoder":
                mse_fig = plt.figure(figsize=(7, 7))
                ax_mse = mse_fig.add_subplot()
                mse = list(data["gating_results"])
            col_x = list(map(float, dataframe[:, indexes[0]]))
            col_y = list(map(float, dataframe[:, indexes[1]]))
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)
            colors = cmap(np.linspace(0, 1, len(labels_uniq)))
            for _, (label, color) in enumerate(zip(labels_uniq, colors), 1):
                indexes_plot = np.where(labels == label)
                axis_x = np.take(col_x, indexes_plot, axis=0)
                axis_y = np.take(col_y, indexes_plot, axis=0)
                if settings.vis_dims == 3:
                    col_z = list(map(float, dataframe[:, indexes[2]]))
                    axis_z = np.take(col_z, indexes_plot, axis=0)
                    ax.scatter3D(
                        axis_x, axis_y, axis_z, c=color, label=label, norm=norm
                    )
                else:
                    ax.scatter(axis_x, axis_y, c=color, label=label)
                if settings.gating_type == "Autoencoder":
                    mse_axis_y = np.take(mse, indexes_plot, axis=0)
                    ax_mse.scatter(indexes_plot, mse_axis_y, c=color, label=label)
            ax.set_xlabel(channels_use[0])
            ax.set_ylabel(channels_use[1])
            if settings.vis_dims == 3:
                ax.set_zlabel(channels_use[2])
                ax.set_title("3D Scatter Plot of " + file)
            else:
                ax.set_title("2D Scatter Plot of " + file)
            ax.legend()
            fig.savefig(self.output_path + file + "_" + "predictions.png")
            if settings.gating_type == "Autoencoder":
                ax_mse.set_xlabel("Index")
                ax_mse.set_ylabel("MSE")
                ax_mse.axhline(y=settings.mse_threshold, color="r", linestyle="-")
                ax_mse.set_title("Reconstruction Error for " + file)
                ax_mse.legend()
                mse_fig.savefig(self.output_path + file + "_" + "mse.png")
            plt.close()

    def diagnostics(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        predicted_labels_probs: np.ndarray,
        gating_results: np.ndarray,
        mse_threshold: float,
    ) -> np.ndarray:
        """
        Creates plots that are used to access the performance of the model. Plots include ROC curve, precision-recall
        pie chart of overall accuracy, aggregated confusion matrix, and MSE histogram.
        Args:
            true_labels (np.ndarray): True labels.
            predicted_labels (np.ndarray): Predicted labels.
            predicted_labels_probs (np.ndarray): Probabilities of predicted labels.
            gating_results (np.ndarray): Reconstruction errors.
            mse_threshold (float): Threshold for the reconstruction error.
        Returns:
            labels_compared (np.ndarray): Array of correct/incorrect labels.
        Raises:
            ValueError: If the number of true labels and predicted labels are not equal.
        See Also:
            :meth:`__pie`.
            :meth:`__roc`.
            :meth:`__precision_recall`.
            :meth:`__aggregated_matrix`.
            :meth:`__mse`.
        """
        if np.unique(true_labels).shape[0] != np.unique(predicted_labels).shape[0]:
            raise ValueError(
                "Number of classes in true and predicted labels are not equal"
            )
        true_labels_binarized, predicted_labels_probs = drop_blanks(
            true_labels, predicted_labels_probs
        )
        self.classes = np.unique(true_labels_binarized)
        self.n_classes = self.classes.shape[0]
        true_labels_binarized = label_binarize(
            true_labels_binarized, classes=self.classes[self.classes != "Blank"]
        )
        self.classes = {
            i: bacteria_class for i, bacteria_class in enumerate(self.classes)
        }
        labels_compared = self.__pie(true_labels, predicted_labels)
        self.__roc(true_labels_binarized, predicted_labels_probs)
        self.__precision_recall(true_labels_binarized, predicted_labels_probs)
        self.__confusion_matrix(true_labels, predicted_labels)
        if type(gating_results[0]) == float:
            self.__mse_scatter(gating_results, true_labels, mse_threshold)
        return labels_compared

    def __pie(
        self, true_labels: np.ndarray, predicted_labels: np.ndarray
    ) -> np.ndarray:
        """
        Creates pie chart of correct/incorrect labels.
        Args:
            true_labels (np.ndarray): True labels.
            predicted_labels (np.ndarray): Predicted labels.
        Returns:
            labels_compared (np.ndarray): Array of correct/incorrect labels.
        """
        labels = []
        for i in range(0, len(true_labels)):
            labels.append("Correct") if true_labels[i] == predicted_labels[
                i
            ] else labels.append("Incorrect")
        labels = np.array(labels)
        _, count_labels = np.unique(np.asarray(labels), return_counts=True)
        plt.pie(count_labels, labels=["Correct", "Incorrect"], autopct="%.0f%%")
        plt.title("Pie Chart of Prediction Accuracy")
        plt.savefig(self.output_path + "pie_no_blanks.png")
        plt.close()
        return labels

    def __roc(
        self, true_labels: np.ndarray, predicted_labels_probs: np.ndarray
    ) -> None:
        """
        Creates plot of ROC curves.
        Args:
            true_labels (np.ndarray): True labels.
            predicted_labels_probs (np.ndarray): Probabilities of predicted labels.
        Returns:
            None.
        """
        if self.n_classes > 2:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.n_classes):
                fpr[i], tpr[i], _ = roc_curve(
                    true_labels[:, i], predicted_labels_probs[:, i]
                )
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= self.n_classes
            fpr["micro"], tpr["micro"], _ = roc_curve(
                true_labels.ravel(), predicted_labels_probs.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            plt.figure()
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(
                    roc_auc["micro"]
                ),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )
            plt.plot(
                fpr["macro"],
                tpr["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(
                    roc_auc["macro"]
                ),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            for i, color in zip(range(self.n_classes), self.colors):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=2,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(
                        self.classes[i], roc_auc[i]
                    ),
                )
            plt.plot([0, 1], [0, 1], "k--", lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves")
            plt.legend(loc="lower right")
        else:
            fpr, tpr, _ = roc_curve(true_labels[:, 0], predicted_labels_probs[:, 1])
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.savefig(self.output_path + "roc.png")
        plt.close()

    def __precision_recall(
        self, true_labels: np.ndarray, predicted_labels_probs: np.ndarray
    ) -> None:
        """
        Creates plot of precision-recall curves.
        Args:
            true_labels (np.ndarray): True labels.
            predicted_labels_probs (np.ndarray): Probabilities of predicted labels.
        Returns:
            None.
        """
        if self.n_classes > 2:
            precision = dict()
            recall = dict()
            average_precision = dict()
            for i in range(self.n_classes):
                precision[i], recall[i], _ = precision_recall_curve(
                    true_labels[:, i], predicted_labels_probs[:, i]
                )
                average_precision[i] = average_precision_score(
                    true_labels[:, i], predicted_labels_probs[:, i]
                )
            _, ax = plt.subplots(figsize=(7, 8))
            f_scores = np.linspace(0.2, 0.8, num=4)
            iso_curves = None
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (iso_curves,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
            display = None
            for i, color in zip(range(self.n_classes), self.colors):
                display = PrecisionRecallDisplay(
                    recall=recall[i],
                    precision=precision[i],
                    average_precision=average_precision[i],
                )
                display.plot(
                    ax=ax, name=f"Precision-recall for {self.classes[i]}", color=color
                )
            handles, labels = display.ax_.get_legend_handles_labels()
            handles.extend([iso_curves])
            labels.extend(["Iso-f1 curves"])
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.legend(handles=handles, labels=labels, loc="lower right")
            ax.set_title("Precision-Recall Curves")
        else:
            precision, recall, _ = precision_recall_curve(
                true_labels, predicted_labels_probs[:, 1]
            )
            PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        plt.savefig(self.output_path + "precision_recall.png")
        plt.close()

    def __confusion_matrix(
        self, true_labels: np.ndarray, predicted_labels: np.ndarray
    ) -> None:
        """
        Creates aggregated confusion matrix.
        Args:
            true_labels (np.ndarray): True labels.
            predicted_labels (np.ndarray): Predicted labels.
        Returns:
            None.
        """
        true_labels, true_labels_counts = np.unique(true_labels, return_inverse=True)
        _, predicted_labels_counts = np.unique(predicted_labels, return_inverse=True)
        cm = confusion_matrix(true_labels_counts, predicted_labels_counts)
        plt.figure()
        plot_confusion_matrix(
            conf_mat=cm, class_names=true_labels, figsize=(7, 8), show_normed="true"
        )
        plt.tight_layout()
        plt.title("Confusion Matrix")
        plt.savefig(self.output_path + "confusion_matrix.png")
        plt.close()

    def __mse_scatter(
        self, mse: np.ndarray, true_labels: np.ndarray, mse_threshold: float
    ) -> None:
        """
        Creates scatter plot of MSE values.
        Args:
            mse (np.ndarray): MSE values.
            true_labels (np.ndarray): True labels.
            mse_threshold (float): Threshold for MSE values.
        Returns:
            None.
        """
        plt.figure(figsize=(7, 7))
        uniq_labels, _ = np.unique(true_labels, return_inverse=True)
        for label, color in zip(uniq_labels, self.colors):
            mse = list(mse)
            indexes_plot = np.where(true_labels == label)
            mse_axis_y = np.take(mse, indexes_plot, axis=0)
            plt.scatter(indexes_plot, mse_axis_y, c=color, label=label)
        plt.axhline(y=mse_threshold, color="r", linestyle="-")
        plt.xlabel("Index")
        plt.ylabel("MSE")
        plt.title("MSE Scatter")
        plt.legend()
        plt.savefig(self.output_path + "mse_scatter.png")
        plt.close()
