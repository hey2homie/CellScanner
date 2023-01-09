import warnings
from typing import Tuple

import tensorflow as tf
import numpy as np
import pandas as pd
import fcsparser

from sklearn.neural_network import MLPClassifier

from utilities.settings import Settings, ModelsInfo


class FilePreparation:
    """
    FilePreparation class is used for initial datasets preprocessing, including the following steps: conversion,
    scaling, columns dropping, gating and formatting labels. The output, which is a dictionary containing varies
    dataframes and arrays, is used either for input for the models during prediction steps or as the source for creating
    TensorFlow datasets.

    Attributes:
    ----------
    files_list: list
        List, containing paths to the files to be processed.
    settings: Settings
        Settings object, containing the settings for the current run.
    models_info: ModelsInfo
        ModelsInfo object, containing the model's metadata.
    data: dict
        Dictionary, containing the dataframes and arrays, which are the output of the preprocessing steps.
    training_cls: bool
        Boolean indicating whether the current run is training classifier or prediction.
    """

    def __init__(self, files: list, settings: Settings, models_info: ModelsInfo, training_cls: bool = False) -> None:
        """
        Args:
            files (list): List containing paths to the files to be processed.
            settings (Settings): Settings object, containing the settings for the current run.
            models_info (ModelsInfo): ModelsInfo object, containing the model's metadata.
            training_cls (bool): Optional; boolean indicating whether the current run is training or prediction.
        Returns:
            None.
        """
        self.files_list = files
        self.settings = settings
        self.models_info = models_info
        self.data = {}
        self.training_cls = training_cls
        self.gating = True
        self.classifier_gating = None

    def __clean_files(self) -> None:
        """
        Removes the files, which are not in the FCS format.
        Returns:
            None
        """
        files_list = []
        for file in self.files_list:
            extension = file.split(".")[-1]
            if extension in ["fcs", "csv", "tsv", "xlsx"]:
                files_list.append(file)
        self.files_list = files_list

    @staticmethod
    def __convert(file: str) -> pd.DataFrame:
        """
        Static method. Converts the input file to a pandas dataframe. Accepts .fcs, .csv, .tsv and .xlsx files.
        Args:
            file (str): Path to the file to be converted.
        Returns:
            dataframe (pd.DataFrame): Pandas' dataframe.
        """
        extension = file.split(".")[-1]
        if extension == "fcs":
            _, data = fcsparser.parse(file, meta_data_only=False, reformat_meta=False)
            return data
        elif extension == "csv":
            return pd.read_csv(file)
        elif extension == "tsv":
            return pd.read_csv(file, sep="\t")
        elif extension == "xlsx":
            return pd.read_excel(file)

    def __drop_columns(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Drops the columns. Columns are dropped based on the settings in case of preparing data for autoencoder training.
        In all other cases, the columns are dropped based on the currently selected autoencoder metadata.
        Args:
            dataframe (pd.DataFrame): Dataframe to be processed.
        Returns:
            dataframe (pd.DataFrame): Resulting dataframe.
        Raises:
            KeyError: An error occurred accessing the dataframe's columns.
        """
        if self.gating:
            self.models_info.autoencoder_name = self.settings.autoencoder
            cols_to_drop = self.models_info.get_features_ae()
        else:
            cols_to_drop = self.settings.cols_to_drop_accuri if self.settings.fc_type == "Accuri" else \
                self.settings.cols_to_drop_cytoflex
        try:
            dataframe = dataframe[cols_to_drop] if self.gating else dataframe.drop(cols_to_drop, axis=1)
            features = np.array(dataframe.columns)
            return dataframe, features
        except KeyError:
            raise KeyError(f"Columns to drop are not present in the dataframe. Columns to drop: {cols_to_drop}")

    @staticmethod
    def __scale(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Static method. Scales the input dataframe using the arcsinh
        Args:
            dataframe (pd.DataFrame): Dataframe to be scaled.
        Returns:
            dataframe (pd.DataFrame): Scaled dataframe.
        """
        for column in dataframe.select_dtypes(include=[np.number]).columns:
            with np.errstate(all="ignore"):
                dataframe[column] = np.arcsinh(dataframe[column].values)
        return dataframe

    def __gate(self, dataframe: pd.DataFrame, file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs gating on the input dataframe. Can be done either by using the autoencoder or by using the
        implementation from the previous version of CellScanner. In case of using the autoencoder, the reconstruction
        error is calculated, which is further used together with the MSE threshold to classify cell either as actual
        cell, or some sort of artifact. When training the autoencoder, the whole step is skipped.
        Args:
            dataframe (pd.DataFrame): Dataframe to be gated.
            file (str): Path to the file, which is currently being processed.
        Returns:
            data (np.ndarray): Passed dataframe but converted to np.ndarray.
            mse (np.ndarray): Reconstruction error for each observation.
        """
        mse = None
        if self.settings.gating_type == "Autoencoder" and self.gating:
            from utilities.classification_utils import AutoEncoder
            self.models_info.autoencoder_name = self.settings.autoencoder
            autoencoder = AutoEncoder(settings=self.settings, model_info=self.models_info, model_type="ae",
                                      name=self.settings.autoencoder)
            autoencoder = autoencoder.get_model()
            predicted = autoencoder.predict(dataframe)
            mse = np.mean(np.power(dataframe - predicted, 2), axis=1)
            if self.training_cls:
                dataframe = dataframe[mse < self.settings.mse_threshold].values
        elif self.settings.gating_type == "Machine":
            raise NotImplementedError
        return np.array(dataframe), mse

    @staticmethod
    def __add_labels(file: str, length: int, gating: bool = False) -> np.ndarray:
        """
        Static method. Adds labels to the input dataframe. Labels are added based on the file name. Files are expected
        to be named in the following formats: "Name-name_...fcs" or "Name_...fcs".
        Args:
            file (str): Path to the file.
            length (int): Length of the dataframe.
        Returns:
            labels (np.ndarray): Array containing the label for each observation.
        """
        split = file.split("/")[-1].split("-")[0].split("_")
        labels = []
        for row in range(0, length):
            try:
                name = split[0] + " " + split[1]
            except IndexError:
                name = split[0]
            labels.append(name)
        return np.array(labels)

    def get_prepared_inputs(self) -> dict:
        """
        Runs the file preprocessing pipeline.
        Returns:
            data (dict): dictionary containing the preprocessed data, mse, labels, and features.
        """
        self.__clean_files()
        for file in self.files_list:
            self.data[file] = self.__convert(file=file)
            try:
                self.data[file], features = self.__drop_columns(self.data[file])
            except TypeError:
                warnings.warn(f"The {file} comes from a different flow cytometer", category=UserWarning)
                self.data.pop(file)
                pass
            self.data[file] = self.__scale(self.data[file])
            self.data[file], mse = self.__gate(self.data[file], file=file)
            labels = self.__add_labels(file=file, length=self.data[file].shape[0])
            self.data[file] = {"data": self.data[file], "mse": mse, "labels": labels, "features": features}
        return self.data

    def get_aggregated(self) -> dict:
        """
        Aggregates the data from all the files into one dataframe.
        Returns:
            data (dict): Dictionary containing the aggregated data, mse, labels, features, and file names.
        See Also:
            :func:`get_prepared_inputs`.
        """
        self.get_prepared_inputs()
        data, labels, mse, features = [], [], [], []
        for key, value in self.data.items():
            data.append(value["data"])
            mse.append(value["mse"])
            labels.append(value["labels"])
            features.append(value["features"])
        features = list(features[0])
        try:
            data, mse, labels = np.concatenate(data, axis=0), np.concatenate(mse, axis=0), \
                                np.concatenate(labels, axis=0)
        except ValueError:
            data, labels = np.concatenate(data, axis=0), np.concatenate(labels, axis=0)
        return {"data": data, "mse": mse, "labels": labels, "features": features, "files": self.files_list}


class DataPreparation:
    """
    DataPreparation class is used to convert preprocessed files into TensorFlow datasets.

    Attributes:
    ----------
    dataframe: np.ndarray
        Numpy array containing the data.
    labels: np.ndarray = None
        Optional; Numpy array containing the labels. Not needed for training autoencoder.
    batch_size: int = 256
        Optional; Batch size for the dataset.
    """

    def __init__(self, dataframe: np.ndarray, labels: np.ndarray = None, batch_size: int = 256) -> None:
        """
        Args:
            dataframe (np.ndarray): Numpy array containing the data.
            labels (np.ndarray): Optional; Numpy array containing the labels.
            batch_size (int): Optional; Batch size for the dataset.
        Returns:
            None.
        """
        self.dataframe = dataframe
        self.labels = labels
        self.labels_ints = None
        self.batch_size = batch_size

    def __map_labels(self) -> None:
        """
        Maps the labels to integers and ecnodes them using one-hot encoding.
        Returns:
            None.
        """
        self.labels, self.labels_ints = np.unique(self.labels, return_inverse=True)
        self.labels_ints = tf.convert_to_tensor(self.labels_ints, dtype=tf.float32)
        self.labels_ints = tf.keras.utils.to_categorical(self.labels_ints, num_classes=len(self.labels))

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Creates the training and validation datasets.
        Returns:
            train_dataset (tf.data.Dataset): Training dataset.
            val_dataset (tf.data.Dataset): Validation dataset.
        """
        if self.labels is not None:
            self.__map_labels()
            dataset = tf.data.Dataset.from_tensor_slices((self.dataframe, self.labels_ints))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.dataframe, self.dataframe))
        tf.keras.utils.set_random_seed(40)
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
        """
        Returns the labels map.
        Returns:
            labels_map (dict): Dictionary containing pairs index:label.
        See Also:
            :func:`__map_labels`
        """
        return {index: value for index, value in enumerate(self.labels)}
