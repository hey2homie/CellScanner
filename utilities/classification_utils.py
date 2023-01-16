import os
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as k
from keras.models import Model, Sequential
from keras.layers import InputLayer, Lambda, BatchNormalization, Conv1D, MaxPooling1D, Dense, Activation, Flatten, \
    Dropout
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, Precision, Recall, TruePositives, FalsePositives, BinaryAccuracy, \
    TrueNegatives, FalseNegatives
from keras.activations import relu, elu
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .data_preparation import FilePreparation, DataPreparation
from .helpers import create_output_dir, match_blanks_to_predictions
from .visualizations import MplVisualization, UmapVisualization
from .settings import Settings, ModelsInfo


class AppModels(ABC):
    """
    Base class for both classifier and autoencoder models.

    Attributes:
    ----------
    settings: Settings
        Settings object.
    model_info: ModelsInfo
        ModelsInfo object.
    model_type: str
        Type of the model. Either "classifier" or "autoencoder".
    name: str = None
        Optional; Name of the model.
    files: list = None
        Optional; List of files used for training.
    training: bool = False
        Optional; Whether the model is being trained or not.
    """

    def __init__(self, settings: Settings, model_info: ModelsInfo, model_type: str, name: str = None,
                 files: list = None, training_cls: bool = False):
        """
        Args:
            settings (Settings): Settings object.
            model_info (ModelsInfo): ModelsInfo object.
            model_type (str): Type of the model. Either "classifier" or "autoencoder".
            name (str, optional): Optional; Name of the model. Defaults to None.
            files (list, optional): Optional; List of files used for training. Defaults to None.
            training_cls (bool, optional): Optional; Whether the model is being trained or not. Defaults to False.
        Returns:
            None.
        """
        self.name = name
        self.settings = settings
        self.model_info = model_info
        self.model_type = model_type
        self.file_prep = self.__pre_process(files, training=training_cls)
        self.model = None

    def __build_model(self) -> None:
        """
        Constructs the model object. Configuration of the model is depended on the type.
        Returns:
            None.
        """
        if self.model_type == "classifier":
            num_features = self.model_info.get_features_shape_classifier()
            num_classes = self.model_info.get_labels_shape()
            if self.settings.mlp:
                self.model = Sequential([
                    InputLayer(input_shape=num_features),
                    Activation(activation=relu),
                    Dense(250),
                    Activation(activation=relu),
                    Dense(num_classes, activation="softmax")
                ])
            else:
                self.model = Sequential([
                    InputLayer(input_shape=num_features),
                    Lambda(lambda x: tf.expand_dims(x, -1)),
                    BatchNormalization(),
                    Conv1D(filters=20, kernel_size=5, padding="valid"),
                    MaxPooling1D(),
                    Activation(activation=elu),
                    Dense(50, kernel_initializer="he_uniform"),
                    Activation(activation=elu),
                    Dense(150, kernel_initializer="he_uniform"),
                    Activation(activation=elu),
                    Dense(300, kernel_initializer="he_uniform"),
                    Activation(activation=elu),
                    Dropout(0.35),
                    Dense(500, kernel_initializer="he_uniform"),
                    Activation(activation=elu),
                    Dropout(0.5),
                    Dense(150, kernel_initializer="he_uniform"),
                    Activation(activation=elu),
                    Dense(30, kernel_initializer="he_uniform"),
                    Activation(activation=elu),
                    Flatten(),
                    Dense(num_classes, activation="softmax", kernel_initializer="glorot_uniform")
                ])
        else:
            num_features = self.model_info.get_features_shape_ae()
            encoder = Sequential([
                InputLayer(input_shape=num_features),
                Lambda(lambda x: tf.expand_dims(x, -1)),
                BatchNormalization(),
                Conv1D(filters=20, kernel_size=5, padding="valid"),
                MaxPooling1D(),
                Activation(activation=elu),
                BatchNormalization(),
                Dense(units=15, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Dense(units=7, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Flatten(),
                Dense(units=5, kernel_initializer="he_uniform"),
                Activation(activation=elu),
            ])
            decoder = Sequential([
                Dense(units=7, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Dense(units=10, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Dense(units=num_features, kernel_initializer="he_uniform")
            ])
            self.model = Model(inputs=encoder.input, outputs=decoder(encoder.output))

    def compile_model(self) -> None:
        """
        Compiles the model. Compiling parameters are depended on the type of the model.
        Returns:
            None.
        """
        self.__build_model()
        metrics = [TruePositives(), FalsePositives(), TrueNegatives(), FalseNegatives()]
        if self.model_type == "classifier":
            num_classes = self.model_info.get_labels_shape()
            if num_classes > 2:
                loss = CategoricalCrossentropy(from_logits=False)
                metrics.extend([CategoricalAccuracy(), Precision(), Recall()])
            else:

                def precision_metric(y_true, y_pred):
                    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
                    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
                    precision = true_positives / (predicted_positives + k.epsilon())
                    return precision

                def recall_metric(y_true, y_pred):
                    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
                    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
                    recall = true_positives / (possible_positives + k.epsilon())
                    return recall

                loss = BinaryCrossentropy(from_logits=False)
                metrics.extend([BinaryAccuracy(), precision_metric, recall_metric])
            self.model.compile(optimizer=Adam(float(self.settings.lr)), loss=loss, metrics=metrics)
        else:
            self.model.compile(optimizer=Adam(1e-3), loss="mse")

    def __pre_process(self, files: list, training: bool = False) -> FilePreparation:
        """
        Instantiates the FilePreparation object with the given files.
        Args:
            files (list): List of files used for training.
            training (bool, optional): Optional; Whether the model is being trained or not. Defaults to False.
        Returns:
            FilePreparation: FilePreparation object.
        """
        return FilePreparation(files=files, settings=self.settings, models_info=self.model_info, training_cls=training)

    def create_training_files(self, dataframe: np.ndarray, labels: np.ndarray = None) -> \
            Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[tf.data.Dataset, tf.data.Dataset, dict]:
        """
        Creates training files from the given dataframe and labels.
        Args:
            dataframe (np.ndarray): Pre-processed dataframe.
            labels (np.ndarray, optional): Optional; Labels for the dataframe. Defaults to None. Used only for
                classifier.
        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[tf.data.Dataset, tf.data.Dataset, dict]: In case of
                training classifier training and validation datasets or tuple of training, validation datasets and
                labels map otherwise.
        """
        data_preparation = DataPreparation(dataframe=dataframe, labels=labels, batch_size=self.settings.num_batches)
        training_set, test_set = data_preparation.create_datasets()
        if self.model_type == "classifier":
            labels_map = data_preparation.get_labels_map()
            return training_set, test_set, labels_map
        else:
            return training_set, test_set

    def train_model(self, name: str, training_set: tf.data.Dataset, test_set: tf.data.Dataset,
                    scheduler: list = None) -> None:
        """
        Runs model training. Saves the model and training history to the directory specified in settings object.
        Args:
            name (str): Name of the model, used for saving.
            training_set (tf.data.Dataset): Training dataset.
            test_set (tf.data.Dataset): Validation dataset.
            scheduler (list, optional): Optional; List of learning rate schedulers. Defaults to None.
        Returns:
            None.
        """
        folder = "./classifiers/" if self.model_type == "classifier" else "./autoencoders/"
        if self.model_type == "classifier":
            monitor = "val_categorical_accuracy" if self.model_info.get_labels_shape() > 2 else "val_binary_accuracy"
            checkpoint = ModelCheckpoint(folder + name, monitor=monitor, save_best_only=True)
        else:
            checkpoint = ModelCheckpoint(folder + name, monitor="val_loss", save_best_only=True)
        tf_board = TensorBoard(log_dir="training_logs/" + self.name, histogram_freq=1, write_graph=False,
                               write_images=False, update_freq="epoch")
        callbacks = [checkpoint, tf_board]
        if scheduler:
            callbacks.extend(scheduler)
        history = self.model.fit(training_set, validation_data=test_set, epochs=self.settings.num_epochs,
                                 callbacks=callbacks)
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = "training_logs/" + name + '/history.csv'
        if os.path.isdir("training_logs/") is not True:
            os.mkdir("training_logs/")
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        with open("training_logs/" + name + "/files_used.txt", "w") as file:
            for f in self.file_prep.files_list:
                file.write(f + "\n")

    def get_model(self) -> Model:
        """
        Returns models with loaded weights.
        Returns:
            Model: Model with loaded weights.
        Raises:
            FileNotFoundError: If the model is not found.
        """
        folder = "./classifiers/" if self.model_type == "classifier" else "./autoencoders/"
        if self.model_type == "classifier":
            self.model_info.classifier_name = self.name
        else:
            self.model_info.autoencoder_name = self.name
        self.__build_model()
        if os.path.exists(folder + self.name):
            self.model.load_weights(folder + self.name)
            return self.model
        else:
            raise FileNotFoundError("Weights are not found")

    @abstractmethod
    def run_training(self) -> None:
        """
        Runs the training process.
        Returns:
            None.
        """
        pass


class ClassificationModel(AppModels):
    """
    Class for classification models. Inherits from AppModels.
    """

    def run_training(self) -> None:

        def __time_based_decay(epoch: int) -> float:
            """
            Args:
                epoch (int): Current epoch.
            Returns:
                float: Adjusted learning rate.
            """
            decay = float(self.settings.lr) / int(self.settings.num_epochs)
            return float(self.settings.lr) * 1 / (1 + decay * epoch)

        def __step_decay(epoch: int) -> float:
            """
            Args:
                epoch (int): Current epoch.
            Returns:
                float: Adjusted learning rate.
            """
            drop_rate = 0.5
            epochs_drop = 10.0
            return float(self.settings.lr) * np.power(drop_rate, np.floor(epoch / epochs_drop))

        def __exp_decay(epoch: int) -> float:
            """
            Args:
                epoch (int): Current epoch.
            Returns:
                float: Adjusted learning rate.
            """
            rate = 0.1
            return float(self.settings.lr) * np.exp(-rate * epoch)

        data = self.file_prep.get_aggregated()
        dataframe, labels, files = data["data"], data["labels"], data["files"]
        training_set, test_set, labels_map = self.create_training_files(dataframe, labels)
        num_features, num_classes = dataframe.shape[1], np.unique(labels).shape[0]
        self.model_info.add_classifier(self.name, self.settings.fc_type, labels_map, num_features, num_classes, files,
                                       self.settings.autoencoder)
        scheduler = None
        if self.settings.lr_scheduler == "Time Decay":
            scheduler = [LearningRateScheduler(__time_based_decay)]
        elif self.settings.lr_scheduler == "Step Decay":
            scheduler = [LearningRateScheduler(__step_decay)]
        elif self.settings.lr_scheduler == "Exponential Decay":
            scheduler = [LearningRateScheduler(__exp_decay)]
        self.compile_model()
        self.train_model(self.name, training_set, test_set, scheduler=scheduler)
        self.model_info.save_info()

    def __make_predictions(self, dataframe: np.ndarray = None, diagnostics: bool = False) -> dict:
        """
        Makes predictions on the given dataframe.
        Args:
            dataframe (np.ndarray, optional): Optional; Dataframe to make predictions on. Defaults to None.
            diagnostics (bool, optional): Optional; If True, runs model evaluation instead of predictions. In this case,
                no need for input as aggregated dataframe is taken from the FilePreparation object. Defaults to False.
        Returns:
            results (dict): Dictionary containing the following keys: data, mse, true_labels (optional), embeddings
                (optional), labels, probability_pred.
        """
        results = {}
        self.get_model()
        labels_map = self.model_info.get_labels_map()
        labels, probability_pred = [], []
        if diagnostics:
            data = self.file_prep.get_aggregated()
            dataframe = data["data"]
            results["gating_results"], results["true_labels"] = data["gating_results"], data["labels"]
        results["data"] = dataframe
        predictions = self.model.predict(dataframe)
        if self.settings.vis_type == "UMAP":
            umap = UmapVisualization(data=dataframe, num_cores=int(self.settings.num_umap_cores),
                                     dims=int(self.settings.vis_dims))
            embeddings = umap.embeddings
            results["embeddings"] = embeddings
        for _, pred in enumerate(predictions):
            probability_pred.append(tf.get_static_value(pred))
            labels.append(tf.get_static_value(tf.math.argmax(pred)))
        results["labels"] = np.asarray(list(map(lambda x: labels_map[x], labels)))
        results["probability_pred"] = np.asarray(probability_pred)
        results["probability_best"] = np.max(results["probability_pred"], axis=1)
        return results

    def run_classification(self) -> dict:
        """
        Runs the classification process.
        Returns:
            results (dict): Dictionary containing the following keys: data, mse, true_labels (optional), embeddings
                (optional), labels, probability_pred.
        See Also:
            :func:`__make_predictions`.
        """
        outputs = self.file_prep.get_prepared_inputs()
        for key, data in outputs.items():
            outputs[key] = self.__make_predictions(dataframe=data["data"])
            outputs[key]["gating_results"] = data["gating_results"]
        return outputs

    def __diagnostic_plots(self, true_labels: np.ndarray, predicted_labels: np.ndarray,
                           probs: np.ndarray, mse: np.ndarray) -> np.ndarray:
        """
        Launches process of creating diagnostics plots.
        Args:
            true_labels (np.ndarray): Labels from the files.
            predicted_labels (np.ndarray): Predicted labels.
            probs (np.ndarray): Predicted probabilities.
        Returns:
            labels_compared (np.ndarray): Numpy array containing labels of correct/incorrect prediction.
        See Also:
            :func:`.MplVisualization.diagnostics()`.
        """
        output_dir = create_output_dir(path=self.settings.results)
        vis = MplVisualization(output_dir)
        labels_compared = vis.diagnostics(true_labels, predicted_labels, probs, mse, self.settings.mse_threshold)
        return labels_compared

    def run_diagnostics(self) -> dict:
        """
        Runs the model diagnostics process.
        Returns:
            results (dict): Dictionary containing the following keys: data, mse, true_labels (optional), embeddings
                (optional), labels, probability_pred.
        See Also:
            :func:`__make_predictions`.
            :func:`__diagnostic_plots`.
        """
        outputs = self.__make_predictions(diagnostics=True)
        predicted_labels = outputs["labels"]
        true_labels = outputs["true_labels"]
        probs = outputs["probability_pred"]
        gating_results = outputs["gating_results"]
        predicted_labels = match_blanks_to_predictions(predicted_labels, self.settings.gating_type, gating_results,
                                                       self.settings.mse_threshold)
        outputs["labels_compared"] = self.__diagnostic_plots(true_labels, predicted_labels, probs, gating_results)
        return outputs


class AutoEncoder(AppModels):
    """
    Class for the AutoEncoder model.
    """

    def __get_clusters(self) -> Tuple[np.ndarray, np.ndarray, dict, list]:
        """
        Gets the clusters from the KMeans algorithm.
        Returns:
            dataframe (np.ndarray): Numpy array of the data.
            y_predicted (np.ndarray): Numpy array containing cluster label for the observations.
            clusters_blank_count (dict): Dictionary containing the number of blank per clusters.
            features (list): List of features.
        """
        data = self.file_prep.get_aggregated()
        dataframe, labels, features = data["data"], data["labels"], data["features"]
        print("Size of dataframe before clustering: {}".format(dataframe.shape))
        kmeans = KMeans(n_clusters=self.settings.number_of_clusters)
        y_predicted = kmeans.fit_predict(dataframe)
        clusters_content_labels = {}
        for i in np.unique(y_predicted):
            if len(y_predicted[y_predicted == i]) > 5:
                clusters_content_labels[i] = np.take(labels, np.where(y_predicted == i))
        clusters_blank_count = {}
        for key, value in clusters_content_labels.items():
            types, counts = np.unique(value, return_counts=True)
            types = np.char.lower(types)
            count_blank = np.take(counts, np.where(types == "blank")).sum()
            clusters_blank_count[key] = np.round((count_blank / counts.sum()) * 100, 2)
        print("Clusters blank count: {}".format(clusters_blank_count))
        return dataframe, y_predicted, clusters_blank_count, features

    def __remove_blank_clusters(self, dataframe: np.ndarray, y_predicted: np.ndarray, clusters_blank_count: dict) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Static method. Removes the clusters which contain more than 10% of blank samples.
        Args:
            dataframe (np.ndarray): Numpy array of the data.
            y_predicted (np.ndarray): Numpy array containing clusters number for the observations.
            clusters_blank_count (dict): Dictionary containing the number of blank per clusters.
        Returns:
            dataframe (np.ndarray): Numpy array of the data.
            y_predicted (np.ndarray): Numpy array containing cluster label for the observations.
            remaining_clusters (np.ndarray): Numpy array containing remaining clusters labels.
        """
        indexes = []
        for key, value in clusters_blank_count.items():
            if clusters_blank_count[key] > self.settings.blank_threshold:
                indexes.extend(np.where(y_predicted == key)[0].tolist())
        dataframe = np.delete(dataframe, indexes, axis=0)
        y_predicted = np.delete(y_predicted, indexes, axis=0)
        remaining_clusters = np.unique(y_predicted)
        print("Size of dataframe after removing blank clusters: {}".format(dataframe.shape))
        return dataframe, y_predicted, remaining_clusters

    @staticmethod
    def __remove_outliers(dataframe: np.ndarray, y_predicted: np.ndarray, remaining_clusters: np.ndarray) -> np.ndarray:
        """
        Static method. Removes the outliers from the data using Gaussian Mixtures.
        Args:
            dataframe (np.ndarray): Numpy array of the data.
            y_predicted (np.ndarray): Numpy array containing clusters number for the observations.
            remaining_clusters (np.ndarray): Array with the labels of the remaining clusters.
        Returns:
            data_clean (np.ndarray): Numpy array of the data without outliers, which is used to train the model.
        """
        clusters: list
        clusters = remaining_clusters.tolist()
        for cluster in remaining_clusters:
            if dataframe[np.where(y_predicted == cluster)].shape[0] < 5:
                clusters.remove(cluster)
        gm_per_cluster = [
            GaussianMixture(n_components=1, n_init=10, random_state=42).fit(dataframe[np.where(y_predicted == cluster)])
            for cluster in clusters]
        data_clustered = {}
        for i in clusters:
            data_clustered[i] = np.take(dataframe, np.where(y_predicted == i), axis=0)[0]
        anomalies_per_cluster = {}
        data_clustered_clean = {}
        count = 0
        for i in clusters:
            gm_per_cluster: [GaussianMixture]
            densities = gm_per_cluster[count].score_samples(data_clustered[i])
            density_threshold = np.percentile(densities, 4)
            data_clustered_clean[i] = data_clustered[i][densities > density_threshold]
            anomalies_per_cluster[i] = data_clustered[i][densities < density_threshold]
            count += 1
        firs_key = list(data_clustered_clean.keys())[0]
        data_clean = data_clustered_clean[firs_key]
        for key, values in data_clustered_clean.items():
            if key != firs_key:
                data_clean = np.append(data_clean, values, axis=0)
        print("Size of dataframe after removing outliers: {}".format(data_clean.shape))
        return data_clean

    def run_training(self) -> None:
        self.file_prep.gating = False
        dataframe, y_predicted, clusters_blank_count, columns = self.__get_clusters()
        dataframe, y_predicted, remaining_clusters = self.__remove_blank_clusters(dataframe, y_predicted,
                                                                                  clusters_blank_count)
        data_clean = self.__remove_outliers(dataframe, y_predicted, remaining_clusters)
        feature_shape = data_clean.shape[1]
        self.model_info.add_autoencoder(name=self.name, fc_type=self.settings.fc_type, features=columns,
                                        num_features=feature_shape)
        training_set, test_set = self.create_training_files(data_clean)
        self.compile_model()
        self.train_model(self.name, training_set, test_set)
        self.model_info.save_info()
