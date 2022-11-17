import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import InputLayer, Lambda, BatchNormalization, Conv1D, MaxPooling1D, Dense, Activation, Flatten, \
    Dropout
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy, Precision, TruePositives, FalsePositives
from keras.activations import elu
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from utilities.data_preparation import FilePreparation, DataPreparation
from utilities.visualizations import MplVisualization, UmapVisualization
from utilities.settings import Settings, SettingsOptions, ModelsInfo

import matplotlib.pyplot as plt


def set_tf_hardware(hardware: str) -> None:
    try:
        tf.config.set_visible_devices([], hardware)
    except ValueError:
        raise Exception(hardware + "not available")
    except RuntimeError:
        raise Exception("TensorFlow is already running")


class AppModels(ABC):

    def __init__(self, settings: Settings, model_info: ModelsInfo, model_type: str,  name: str = None,
                 files: list = None, training: bool = False):
        self.name = name
        self.settings = settings
        self.model_info = model_info
        self.model_type = model_type
        self.files = files
        self.file_prep = self.__pre_process(files, training=training)
        self.model = None

    def __build_model(self) -> None:
        if self.model_type == "classifier":
            num_features = self.model_info.get_features_shape_classifier()
            num_classes = self.model_info.get_labels_shape()
            self.model = Sequential([
                InputLayer(input_shape=num_features),
                BatchNormalization(),
                Dense(100, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Dense(300, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Dropout(rate=0.2),
                Dense(500, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Dropout(rate=0.5),
                Dense(300, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Dropout(rate=0.3),
                Dense(100, kernel_initializer="he_uniform"),
                Activation(activation=elu),
                Dense(num_classes, activation="relu", kernel_initializer="he_uniform")
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
        self.__build_model()
        if self.model_type == "classifier":
            self.model.compile(optimizer=Adam(float(self.settings.lr)),
                               loss=CategoricalCrossentropy(from_logits=True),
                               metrics=[Accuracy(), Precision(), TruePositives(), FalsePositives()])
        else:
            self.model.compile(optimizer=Adam(1e-3), loss="mse")

    def __pre_process(self, files: list, training: bool = False) -> FilePreparation:
        return FilePreparation(files=files, settings=self.settings, models_info=self.model_info, training=training)

    def create_training_files(self, dataframe: np.ndarray, labels: np.ndarray = None) -> tuple:
        data_preparation = DataPreparation(dataframe=dataframe, labels=labels, batch_size=self.settings.num_batches)
        training_set, test_set = data_preparation.create_datasets()
        if self.model_type == "classifier":
            labels_map = data_preparation.get_labels_map()
            return training_set, test_set, labels_map
        else:
            return training_set, test_set

    def train_model(self, name: str, training_set: tf.data.Dataset, test_set: tf.data.Dataset,
                    scheduler: list = None) -> None:
        folder = "./classifiers/" if self.model_type == "classifier" else "./autoencoders/"
        checkpoint = ModelCheckpoint(folder + name, save_best_only=True)
        tf_board = TensorBoard(log_dir="training_logs/" + self.name, histogram_freq=1, write_graph=False, write_images=False,
                               update_freq="epoch")
        callbacks = [checkpoint, tf_board]
        if scheduler:
            callbacks.extend(scheduler)
        self.model.fit(training_set, validation_data=test_set, epochs=self.settings.num_epochs, callbacks=callbacks)
        with open("training_logs/" + name + "/files_used.txt", "w") as file:
            for f in self.files:
                file.write(f + "\n")

    def get_model(self) -> Model:
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
            raise Exception("Weights file not found")

    @abstractmethod
    def run_training(self):
        pass


class ClassificationModel(AppModels):

    def run_training(self) -> None:

        def __time_based_decay(epoch: int, lr: float) -> float:
            decay = self.settings.lr / self.settings.num_epochs
            # TODO: Fix missing 1 required positional argument: 'lr'
            return lr * 1 / (1 + decay * epoch)

        def __step_decay(epoch: int) -> float:
            drop_rate = 0.5
            epochs_drop = 10.0
            return self.settings.lr * np.pow(drop_rate, np.floor(epoch / epochs_drop))

        def __exp_decay(epoch: int) -> float:
            k = 0.1
            return self.settings.lr * np.exp(-k * epoch)

        data = self.file_prep.get_aggregated()
        dataframe, labels, files = data["data"], data["labels"], data["files"]
        training_set, test_set, labels_map = self.create_training_files(dataframe, labels)
        num_features, num_classes = dataframe.shape[1], np.unique(labels).shape[0]
        self.model_info.add_classifier(self.name, labels_map, num_features, num_classes, files)
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

    def __make_predictions(self, dataframe: np.ndarray = None, mse: np.ndarray = None,
                           diagnostics: bool = False) -> dict:
        results = {"mse": mse}
        self.get_model()
        labels_map = self.model_info.get_labels_map()
        labels, probability_pred = [], []
        if diagnostics:
            data = self.file_prep.get_aggregated()
            dataframe = data["data"]
            results["mse"], results["true_labels"] = data["mse"], data["labels"]
        results["data"] = dataframe
        predictions = self.model.predict(dataframe)
        embeddings = None
        if self.settings.vis_type == "UMAP":
            umap = UmapVisualization(data=dataframe, num_cores=int(self.settings.num_umap_cores),
                                     dims=int(self.settings.vis_dims))
            embeddings = umap.get_embeddings()
            results["embeddings"] = embeddings
        for _, pred in enumerate(predictions):
            probabilities = tf.nn.softmax(pred)
            probability_pred.append(tf.get_static_value(max(probabilities)))
            labels.append(tf.get_static_value(tf.math.argmax(probabilities)))
        results["labels"] = np.asarray(list(map(lambda x: labels_map[x], labels)))
        results["probability_pred"] = np.asarray(probability_pred)
        return results

    def run_classification(self) -> dict:
        outputs = self.file_prep.get_prepared_inputs()
        for key, data in outputs.items():
            outputs[key] = self.__make_predictions(data["data"], data["mse"])
        return outputs

    def __diagnostic_plots(self, true_labels, predicted_labels) -> list:
        vis = MplVisualization(self.settings.results)
        labels = vis.diagnostics(true_labels, predicted_labels)
        return labels

    def run_diagnostics(self) -> dict:
        outputs = self.__make_predictions(diagnostics=True)
        predicted_labels = outputs["labels"]
        true_labels = outputs["true_labels"]
        labels_compared = self.__diagnostic_plots(true_labels, predicted_labels)
        outputs["labels_compared"] = labels_compared
        return outputs


class AutoEncoder(AppModels):

    def __calculate_num_clusters(self) -> int:
        dataframe = self.file_prep.get_aggregated()["data"]
        gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(dataframe) for k in range(1, 11)]
        aics = np.array([model.aic(dataframe) for model in gms_per_k])
        plt.plot(aics, marker="p")
        plt.show()
        while True:
            try:
                optimal_k = int(input("Enter the number of clusters: "))
                break
            except ValueError:
                print("Please enter a valid number")
        plt.close()
        return optimal_k

    def __get_clusters(self, optimal_k: int) -> tuple:
        data = self.file_prep.get_aggregated()
        dataframe, labels, features = data["data"], data["labels"], data["features"]
        kmeans = KMeans(n_clusters=optimal_k)
        y_predicted = kmeans.fit_predict(dataframe)
        clusters_content_labels = {}
        clusters_content = {}
        for i in np.unique(labels):
            clusters_content_labels[i] = np.take(labels, np.where(labels == i))
            clusters_content[i] = np.take(dataframe, np.where(labels == i))
        clusters_blank_count = {}
        for key, value in clusters_content_labels.items():
            types, counts = np.unique(value, return_counts=True)
            count_blank = np.take(counts, np.where(types == "Blank")).sum()
            clusters_blank_count[key] = np.round((count_blank / counts.sum()) * 100, 2)
        return dataframe, y_predicted, clusters_blank_count, features

    @staticmethod
    def __remove_blank_clusters(dataframe, y_predicted, clusters_blank_count: dict) -> tuple:
        indexes = []
        for key, value in clusters_blank_count.items():
            if clusters_blank_count[key] > 10:
                indexes.extend(np.where(y_predicted == key)[0].tolist())
        dataframe = np.delete(dataframe, indexes, axis=0)
        y_predicted = np.delete(y_predicted, indexes, axis=0)
        remaining_clusters = np.unique(y_predicted)
        return dataframe, y_predicted, remaining_clusters

    @staticmethod
    def __remove_outliers(dataframe, y_predicted, remaining_clusters: np.ndarray, name) -> np.ndarray:
        gm_per_cluster = [
            GaussianMixture(n_components=1, n_init=10, random_state=42).fit(dataframe[np.where(y_predicted == k)])
            for k in remaining_clusters]
        data_clustered = {}
        for i in remaining_clusters:
            data_clustered[i] = np.take(dataframe, np.where(y_predicted == i), axis=0)[0]
        anomalies_per_cluster = {}
        data_clustered_clean = {}
        count = 0
        for i in remaining_clusters:
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
        return data_clean

    def run_training(self) -> None:
        num_clusters = self.__calculate_num_clusters()
        dataframe, y_predicted, clusters_blank_count, columns = self.__get_clusters(optimal_k=num_clusters)
        dataframe, y_predicted, remaining_clusters = self.__remove_blank_clusters(dataframe, y_predicted,
                                                                                  clusters_blank_count)
        data_clean = self.__remove_outliers(dataframe, y_predicted, remaining_clusters, self.name)
        feature_shape = data_clean.shape[1]
        self.model_info.add_autoencoder(name=self.name, fc_type=self.settings.fc_type, features=columns,
                                        num_features=feature_shape)
        training_set, test_set = self.create_training_files(data_clean)
        self.compile_model()
        self.train_model(self.name, training_set, test_set)
        self.model_info.save_info()
