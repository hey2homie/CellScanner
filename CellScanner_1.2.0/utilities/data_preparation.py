import warnings
from typing import Tuple

import tensorflow as tf
import numpy as np
import pandas as pd
import fcsparser

from utilities.settings import Settings, ModelsInfo


class FilePreparation:

    def __init__(self, files: list, settings: Settings, models_info: ModelsInfo, training: bool = False) -> None:
        self.files_list = files
        self.settings = settings
        self.models_info = models_info
        self.data = {}
        self.features = {}
        self.training = training

    @staticmethod
    def __convert(file: str, extension: str) -> pd.DataFrame:
        if extension == "fcs":
            _, data = fcsparser.parse(file, meta_data_only=False, reformat_meta=False)
            return data
        elif extension == "csv":
            return pd.read_csv(file)
        elif extension == "tsv":
            return pd.read_csv(file, sep="\t")
        elif extension == "xlsx":
            return pd.read_excel(file)

    def __drop_columns(self, dataframe: pd.DataFrame) -> tuple or bool:
        if self.training:   # TODO: The case should be only when training autoencoder
            cols_to_drop = self.settings.cols_to_drop_accuri if self.settings.fc_type == "Accuri" else \
                self.settings.cols_to_drop_cytoflex
        else:
            self.models_info.autoencoder_name = self.settings.autoencoder
            cols_to_drop = self.models_info.get_features_ae()
        try:
            dataframe = dataframe[cols_to_drop]
            features = np.array(dataframe.columns)
            return dataframe, features
        except KeyError:
            return False

    @staticmethod
    def __scale(dataframe: pd.DataFrame) -> pd.DataFrame:
        for column in dataframe.select_dtypes(include=[np.number]).columns:
            with np.errstate(divide="ignore"):
                dataframe[column] = np.log10(dataframe[column].values)
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.dropna(inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    def __gate(self, dataframe: pd.DataFrame) -> np.ndarray:
        if self.settings.gating_type == "Autoencoder":
            from utilities.classification_utils import AutoEncoder
            self.models_info.autoencoder_name = self.settings.autoencoder
            autoencoder = AutoEncoder(settings=self.settings, model_info=self.models_info, model_type="ae",
                                      name=self.settings.autoencoder)
            autoencoder = autoencoder.get_model()
            try:
                predicted = autoencoder.predict(dataframe)
            except ValueError:
                raise ValueError("The autoencoder cannot process this file")    # TODO: Print file name
                # Because of the different number of features
            mse = np.log10(np.mean(np.power(dataframe - predicted, 2), axis=1))
            # TODO: Do not remove observations but add a column with the mse values
            # Then, in plotly add a slider to filter out the observations with mse > threshold
            # Though plot of MSE distribution should be there as well
            dataframe = dataframe[mse < self.settings.mse_threshold]
        else:
            raise NotImplementedError
        return np.array(dataframe)

    @staticmethod
    def __add_labels(file: str, dataframe: pd.DataFrame) -> np.ndarray:
        split = file.split("/")[-1].split("-")[0].split("_")
        labels = []
        for row in range(0, dataframe.shape[0]):
            try:
                labels.append(split[0] + " " + split[1])
            except IndexError:
                labels.append(split[0])
        return np.array(labels)

    def get_prepared_inputs(self) -> dict:
        for file in self.files_list:
            extension = file.split(".")[-1]
            if extension not in ["fcs", "csv", "tsv", "xlsx"]:
                pass
            self.data[file] = self.__convert(file=file, extension=extension)
            try:
                self.data[file], self.features[file] = self.__drop_columns(self.data[file])
            except TypeError:
                warnings.warn(f"The {file} comes from a different flow cytometer", category=UserWarning)
                self.data.pop(file)
            self.data[file] = self.__scale(self.data[file])
            self.data[file] = self.__gate(self.data[file])
            self.data[file] = [self.data[file], self.__add_labels(file=file, dataframe=self.data[file])]
        return self.data

    def get_aggregated(self) -> tuple:
        self.get_prepared_inputs()
        features = np.unique(self.features.values())
        data, labels = [], []
        for key, value in self.data.items():
            data.append(value[0])
            labels.append(value[1])
        data, labels = np.array(data)[-1], np.array(labels)[-1]
        return data, labels, features
        # TODO: Do I need features?


class DataPreparation:

    def __init__(self, dataframe: np.ndarray, labels: np.ndarray = None, batch_size: int = 256) -> None:
        self.dataframe = dataframe
        self.labels = labels
        self.labels_ints = None
        self.batch_size = batch_size

    def __map_labels(self) -> None:
        self.labels, self.labels_ints = np.unique(self.labels, return_inverse=True)
        self.labels_ints = tf.convert_to_tensor(self.labels_ints, dtype=tf.float32)

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        if self.labels is not None:
            self.__map_labels()
            dataset = tf.data.Dataset.from_tensor_slices((self.dataframe, self.labels_ints))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.dataframe, self.dataframe))  # Case for autoencoders
        dataset = dataset.shuffle(self.dataframe.shape[0])
        dataset_length = sum(1 for _ in dataset)
        training_length = np.rint(dataset_length * 0.8)
        training_set = dataset.take(training_length)
        test_set = dataset.skip(training_length)
        training_set = training_set.batch(self.batch_size)
        test_set = test_set.batch(self.batch_size)
        training_set = training_set.prefetch(1)
        test_set = test_set.prefetch(1)
        return training_set, test_set

    def get_labels_map(self) -> dict:
        return {index: value for index, value in enumerate(self.labels)}
