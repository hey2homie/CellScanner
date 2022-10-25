import os
from typing import Tuple

import numpy as np

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import InputLayer, Lambda, BatchNormalization, Conv1D, MaxPooling1D, Dense, Activation, Flatten, \
    Dropout
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.activations import elu
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from utilities.data_preparation import FilePreparation, DataPreparation
from utilities.visualizations import UmapVisualization, MplVisualization
from utilities.settings import Settings, ModelsInfo


def set_tf_hardware(hardware: str) -> None:
    """
    Sets the device to run TensorFlow computations. Either CPU or GPU.
    Args:
        hardware (str): String determining whether CPU or GPU will be used as the TensorFlow device.
    Returns:
        None
    """
    try:
        tf.config.set_visible_devices([], hardware)
    except ValueError:
        raise Exception(hardware + "not available")
    except RuntimeError:
        raise Exception("TensorFlow is already running")


class ClassificationModel:
    """
    ClassificationModel class is used to create TensorFlow model, compile it, and load weights once the model have been
    trained.
    """

    def __init__(self, num_features: tuple, num_classes: tuple, fc_type: str, lr: float) -> None:
        """
        Args:
            num_features (tuple): Tuple containing number of features in the dataset used to initialize input layer.
            num_classes (tuple): Tuple containing number of classification files.
        """
        self.feature_shape = num_features[0]
        self.num_classes = num_classes[0]
        self.fc_type = fc_type
        self.lr = lr
        self.model = self.__compile_model()

    def __build_model(self) -> Model:
        """
        Build_model function creates the model architecture and returns resulting Keras model.
        Returns
            keras.Model
        """
        model = Sequential([
            InputLayer(input_shape=self.feature_shape),
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
            Dense(self.num_classes, activation="relu", kernel_initializer="he_uniform")
        ])
        return model

    def __compile_model(self) -> Model:
        """
        Compiles model with the Adam optimizer and learning rate as specified in the settings configuration file.
        Returns:
            model (Model): Compiled model with specified optimizer and learning rate.
        """
        model = self.__build_model()
        opt = Adam(self.lr)
        model.compile(optimizer=opt, loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
        return model

    def get_model(self) -> Model:
        """
        Returns
            self.model (Model): Returns compiled classifier.
        """
        return self.model

    def get_loaded_model(self, weights: str) -> Model:
        """
        Loads model, which is specified in the configuration file.
        Args:
            weights (str, optional): Relative or absolute path to the weights that will be loaded into the model.
        Returns:
            self.model (Model): Returns compiled classifier with loaded weights.
        """
        if os.path.exists(weights):
            self.model.load_weights(weights)
            return self.model
        else:
            raise Exception("Weights file not found")


class ClassificationTraining:
    """
    ClassificationTraining class is used to run training on the provided files.
    """
    # TODO: Tensorboard for visualisations?
    def __init__(self, files: list, model_name: str, settings: Settings, models_info: ModelsInfo) -> None:
        """
        Args:
            files (list): List containing paths to the files that will be used for training.
            model_name (str): Name of the model that will be saved.
            settings (Settings): Settings object containing training parameters.
            models_info (ModelsInfo): ModelsInfo object containing models information.
        """
        self.files = files
        self.model_name = model_name + ".h5"
        self.settings = settings
        self.models_info = models_info
        set_tf_hardware(self.settings.hardware)  # TODO: Consider calling this function at the start of the application
        self.model, self.training_set, self.test_set = self.__prepare_training()

    def __prepare_training(self) -> Tuple[Model, tf.data.Dataset, tf.data.Dataset]:
        """
        Runs file preparation, resulting dataframe is used to create TensorFlow datasets, which are later used in
        training.
        Returns:
            model (Model), training_set (tf.data.Dataset), test_set (tf.data.Dataset): Tuple containing complied
            model, TensorFlow datasets for training and testing.
        """
        file_prep = FilePreparation(files=self.files, settings=self.settings, models_info=self.models_info)
        dataframe = file_prep.get_aggregated()
        labels = file_prep.get_labels()
        labels_shape = np.unique(labels).shape[0]
        data_preparation = DataPreparation(dataframe=dataframe, labels=labels, batch_size=self.settings.num_batches)
        training_set, test_set = data_preparation.create_datasets()
        labels_map = data_preparation.get_labels_map()
        features_shape = None
        for elem in training_set.take(1):
            features_shape = elem[0][1].shape  # TODO: Consider changing this
        model = ClassificationModel(num_classes=labels_shape, num_features=features_shape,
                                    fc_type=self.settings.fc_type, lr=float(self.settings.lr))
        model = model.get_model()
        self.save_model_to_config(labels_map=labels_map, features_shape=tuple(features_shape),
                                  labels_shape=labels_shape)
        return model, training_set, test_set

    def save_model_to_config(self, labels_map: dict, features_shape: tuple, labels_shape: tuple) -> None:
        """
        Saves model information to the dedicated configuration file.
        Args:
            labels_map (dict): Dictionary containing labels and their corresponding indices.
            features_shape (tuple): Shape of the features.
            labels_shape (tuple): Shape of the labels.
        Returns:
            None
        """
        self.models_info.add_classifier(self.model_name, labels_map=labels_map, features_shape=features_shape,
                                        labels_shape=labels_shape)
        self.models_info.save_info()

    def run_training(self) -> None:
        """
        Runs training on the prepared datasets.
        Returns:
            None
        """
        path_to_model = "./classifiers/" + self.model_name
        callbacks_to_use = [ModelCheckpoint(path_to_model, save_best_only=True)]
        if self.settings.lr_scheduler == "Time Based Decay":
            callbacks_to_use.append(LearningRateScheduler(self.__time_based_decay))
        elif self.settings.lr_scheduler == "Step Decay":
            callbacks_to_use.append(LearningRateScheduler(self.__step_decay))
        elif self.settings.lr_scheduler == "Exponential Decay":
            callbacks_to_use.append(LearningRateScheduler(self.__exp_decay))
        self.model.fit(self.training_set, validation_data=self.test_set, epochs=self.settings.num_epochs,
                       callbacks=callbacks_to_use)

    def __time_based_decay(self, epoch: int, lr: float) -> float:
        """
        Args:
            epoch (int): Current epoch
            lr (float): Current learning rate
        Returns:
            decay_rate (float): New learning rate
        """
        decay = self.settings.lr * self.settings.num_epochs
        return lr * 1 / (1 + decay * epoch)

    def __step_decay(self, epoch: int) -> float:
        """
        Args:
            epoch (int): Current epoch
        Returns:
            decay_rate (float): New learning rate
        """
        drop_rate = 0.5
        epochs_drop = 10.0
        return self.settings.lr * np.pow(drop_rate, np.floor(epoch / epochs_drop))

    def __exp_decay(self, epoch: int) -> float:
        """
        Args:
            epoch (int): Current epoch
        Returns:
            decay_rate (float): New learning rate
        """
        k = 0.1
        return self.settings.lr * np.exp(-k * epoch)


class ClassificationResults:
    """
    ClassificationResults class is used to classify input files using previously trained model.
    """
    def __init__(self, files: list, settings: Settings, models_info: ModelsInfo, diagnostics: bool = False) -> None:
        """
        Args:
            files (list): List of strings containing absolute paths to the desired files.
            settings (Settings): Settings object containing settings for making predictions.
            diagnostics (bool, optional): If True, diagnostics is run instead of prediction. Defaults to False.
        """
        self.files = files
        self.models_info = models_info
        self.labels_map = self.models_info.get_labels_map()
        self.settings = settings
        self.diagnostics = diagnostics
        set_tf_hardware(settings.hardware)  # TODO: Consider calling this function at the start of the application
        self.model = ClassificationModel(num_features=models_info.get_features_shape_classifier(),
                                         num_classes=models_info.get_labels_shape(),
                                         fc_type=self.settings.fc_type,
                                         lr=self.settings.lr)
        self.true_labels = {}
        self.outputs = {}
        self.__classification()

    def __classification(self) -> None:
        """
        This functions fills in class attribute outputs, where key is the file and value is the nparray containing
        embeddings from UMAP and corresponding predicted labels.
        Returns:
            None
        """
        file_preparation = FilePreparation(self.files, settings=self.settings, models_info=self.models_info)
        model_name = "./classifiers/" + self.settings.model
        self.model = self.model.get_loaded_model(weights=model_name)
        if self.diagnostics:
            self.outputs["all"] = file_preparation.get_aggregated()
            self.true_labels["all"] = file_preparation.get_labels()
            self.__run_prediction(file="all", file_preparation=file_preparation)
        else:
            for file in self.files:
                self.__run_prediction(file=file, file_preparation=file_preparation)

    def __run_prediction(self, file: str, file_preparation: FilePreparation) -> None:
        """
        Args:
            file (str): String containing absolute path to the file to be classified.
            file_preparation (FilePreparation): FilePreparation object used to prepare file for classification.
        Returns:
            None
        """
        labels = []  # TODO: Change to numpy array
        if not self.diagnostics:
            self.outputs[file] = file_preparation.get_data(file)
        umap = UmapVisualization(self.outputs[file], num_cores=int(self.settings.num_umap_cores), dims=3)
        embeddings = umap.get_embeddings()
        prediction = self.model.predict(self.outputs[file])
        for _, pred in enumerate(prediction):
            probabilities = tf.nn.softmax(pred)
            labels.append(tf.get_static_value(tf.math.argmax(probabilities)))
        labels = np.asarray(list(map(lambda x: self.labels_map[x], labels)))  # TODO: Change to numpy map
        embeddings = np.append(embeddings, labels[:, None], axis=1)
        self.outputs[file] = np.append(self.outputs[file], embeddings, axis=1)

    def get_outputs(self) -> dict:
        """
        Returns:
            self.outputs (dict): Dictionary containing str:nd.array pair, where key is the location of the file and the
            array contains data/UMAP embeddings and predicted labels.
        """
        # TODO: Write outputs to the folder specified in settings.results
        return self.run_diagnostics() if self.diagnostics else self.outputs

    def run_diagnostics(self) -> dict:
        """
        Runs diagnostics on the trained model.
        Returns:
            outputs (dict):
        """
        diagnostics_results = ToolDiagnosticsCalculations(true_labels=self.true_labels, predicted_labels=self.outputs,
                                                          output_path=self.settings.results)
        labels_compared = diagnostics_results.get_misclassified_points()
        outputs = self.get_outputs()
        outputs["all"][:, -1] = labels_compared
        return outputs


class ToolDiagnosticsCalculations:

    def __init__(self, true_labels: dict, predicted_labels: dict, output_path: str) -> None:
        """
        Args:
            true_labels (dict): Dictionary containing str:np.ndarray pair, where key is file and the value is the numpy
            array containing true labels.
            predicted_labels (dict): Dictionary containing str:np.ndarray pair, where key is file and the value is the
            numpy array containing predicted labels.
        Returns:
            None
        """
        self.true_labels = true_labels["all"]
        self.predicted_labels = predicted_labels["all"][:, -1]
        self.output_path = output_path
        self.__calculate_diagnostics()

    def __calculate_diagnostics(self) -> None:
        """

        Returns:
            None
        """
        vis = MplVisualization(output_path=self.output_path)
        vis.diagnostics(true_labels=self.true_labels, predicted_labels=self.predicted_labels)

    def get_misclassified_points(self) -> list:
        """
        Returns:
            labels (list): List of strings containing correctly/incorrectly classified points.
        """
        labels = []  # TODO: Change to numpy array
        for i in range(0, len(self.true_labels)):
            labels.append("Correct") if self.true_labels[i] == self.predicted_labels[i] else labels.append("Incorrect")
        return labels


class AutoEncoder:

    def __init__(self, settings: Settings, models_info: ModelsInfo) -> None:
        self.models_info = models_info
        self.feature_shape = None
        self.autoencoder_name = settings.autoencoder
        self.settings = settings

    def __build_model(self) -> Model:
        encoder = Sequential([
            InputLayer(input_shape=self.feature_shape),
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
            Dense(units=self.feature_shape, kernel_initializer="he_uniform")
        ])
        autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))

        return autoencoder

    def __compile_model(self) -> Model:
        model = self.__build_model()
        model.compile(loss="mse", optimizer=Adam(lr=1e-3))
        return model

    def retrain(self, dataframe: np.ndarray, labels: np.ndarray, name: str, columns: list = None) -> None:
        # TODO: Add method to data preparation to return columns names for them
        gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(dataframe) for k in range(1, 11)]
        aics = np.log10(np.abs(np.array([model.aic(dataframe) for model in gms_per_k])))
        delta_aics = np.diff(aics)
        optimal_k = np.argmax(delta_aics < 0.005) + 1    # TODO: Make this (0.005) hyperparameter adn store in settings
        # TODO: Alright, raising warning window with scores (better with elbow plot) is better
        kmeans = KMeans(n_clusters=optimal_k)
        y_predicted = kmeans.fit_predict(dataframe)
        clusters_content_labels = {}
        clusters_content = {}
        for i in np.unique(y_predicted):
            clusters_content_labels[i] = np.take(labels, np.where(y_predicted == i))
            clusters_content[i] = np.take(dataframe, np.where(y_predicted == i))
        clusters_blank_count = {}
        for key, value in clusters_content_labels.items():
            types, counts = np.unique(value, return_counts=True)
            count_blank = np.take(counts, np.where(types == "Blank")).sum()
            counts_blank_perc = np.round(count_blank / counts * 100, 2)
            clusters_blank_count[key] = {"count_blank %": counts_blank_perc}
        indexes = []
        for key, value in clusters_blank_count.items():
            if clusters_blank_count[key]["count_blank %"] > 10:  # TODO: Make this a hyperparameter
                indexes.extend(np.where(y_predicted == key)[0].tolist())
        dataframe = np.delete(dataframe, indexes, axis=0)
        y_predicted = np.delete(y_predicted, indexes, axis=0)
        remaining_clusters = np.unique(y_predicted)
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
        # TODO: I probably don't need all the code above... soo, come back to it later
        np.save(f"autoencoders/.clean_data/{name}_clean", data_clean)
        data_preparation = DataPreparation(dataframe=data_clean, batch_size=256)
        training_set, test_set = data_preparation.create_datasets()
        self.feature_shape = data_clean.shape[1]
        model = self.__compile_model()
        checkpoint_cb = ModelCheckpoint("./autoencoders/" + name, save_best_only=True)
        model.fit(training_set, validation_data=test_set, epochs=50, callbacks=[checkpoint_cb])
        self.models_info.add_autoencoder(name=name, fc_type=self.settings.fc_type, features=columns,
                                         num_features=self.feature_shape)
        self.models_info.save_info()

    def get_model(self) -> Model:
        self.feature_shape = self.models_info.get_features_shape_ae()
        model = self.__compile_model()
        if os.path.exists("./autoencoders/" + self.autoencoder_name):
            model.load_weights("./autoencoders/" + self.autoencoder_name)
            return model
