from typing import Tuple

import tensorflow as tf
import numpy as np
import pandas as pd
import fcsparser

from utilities.settings import Settings, ModelsInfo


class FilePreparation:
    """
    FilePreparation class is used for the importing and transforming files for the analysis. The main format is .fcs.
    Other supported formats are csv and tsv. Files can be loaded in batch by specifying the files arguments when
    initializing class and later calling get_aggregated() method or processed individually with the get_data() method.
    """

    def __init__(self, files: list, settings: Settings, models_info: ModelsInfo, training: bool = False) -> None:
        """
        Args:
            files (list): List of strings containing absolute path to the desired files.
            settings (Settings): Settings object.
        """
        self.files_list = files
        self.settings = settings
        self.models_info = models_info
        self.fc_type = settings.fc_type
        self.gating_type = settings.gating_type
        self.mse_threshold = settings.mse_threshold
        self.cols_drop = settings.cols_to_drop_accuri if self.fc_type == "Accuri" else settings.cols_to_drop_cytoflex
        self.data = None
        self.aggregated = None
        self.labels = []
        self.features = {}
        self.training = training

    def __check_extension(self):
        """
        Checks if the provided files have correct extension, which can be .fcs, .csv, .tsv, or .xlsx files.
        Returns:
            None
        """
        for file in self.files_list:
            if file.split(".")[-1] not in ["fcs", "csv", "tsv", "xlsx"]:
                self.files_list.remove(file)

    def __convert(self, file: str) -> None:
        """
        Converts given file into pandas dataframe.
        Args:
            file (str): Path to the file.
        """
        extension = file.split(".")[-1]
        if extension == "fcs":
            _, self.data = fcsparser.parse(file, meta_data_only=False, reformat_meta=False)
        elif extension == "csv":
            self.data = pd.read_csv(file)
        elif extension == "tsv":
            self.data = pd.read_csv(file, sep="\t")
        elif extension == "xlsx":
            self.data = pd.read_excel(file)

    def __drop_columns(self, file: str) -> None:
        # TODO: Add extra columns from config
        # TODO: Drop columns according to the autoencoder features
        """
        By default, drops only "time" column. Alternatively, extra columns can be specified in the settings.
        Returns:
            None
        """
        try:
            self.data.drop(self.cols_drop, axis=1, inplace=True)
            self.features[file] = list(self.data.columns)
        except KeyError:
            pass    # TODO: Em, raise issue?

    def __scale(self) -> None:
        """
        Scales each column in the dataframe by log10. If scaling results in the +/- infinity, the whole row is dropped
        from the dataframe. Resets index of the dataframe by the end of scaling.
        Returns:
            None
        """
        for column in self.data.select_dtypes(include=[np.number]).columns:
            with np.errstate(divide="ignore"):
                self.data[column] = np.log10(self.data[column].values)
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def __gate(self) -> None:
        """
        Gates data. The values are taking from the previous implementation of CellScanner.
        Returns:
            None
        """
        if self.gating_type == "Autoencoder":
            from utilities.classification_utils import AutoEncoder
            self.models_info.autoencoder_name = self.settings.autoencoder
            autoencoder = AutoEncoder(settings=self.settings, model_info=self.models_info, model_type="ae",
                                      name=self.settings.autoencoder)
            autoencoder = autoencoder.get_model()
            predicted = autoencoder.predict(self.data)
            mse = np.log10(np.mean(np.power(self.data - predicted, 2), axis=1))
            # TODO: Do not remove observations but add a column with the mse values
            # Then, in plotly add a slider to filter out the observations with mse > threshold
            # Though plot of MSE distribution should be there as well
            self.data = self.data[mse < self.mse_threshold]
        else:
            raise NotImplementedError   # TODO: Implementation of method from the previous version

    def __add_labels(self, file: str) -> None:
        """
        Extracts labels from the file name (assuming they are named correctly) and adds to the labels array, which is
        used during training for creating label maps.
        Args:
            file (str): Path to the file.
        Returns:
            None
        """
        split = file.split("/")[-1].split("-")[0].split("_")
        for row in range(0, self.data.shape[0]):
            try:
                self.labels.append(split[0] + " " + split[1])
            except IndexError:
                self.labels.append(split[0])

    def __run_transformation(self, file: str) -> None:
        """
        Runs transformation pipeline on the individual file.
        Args:
            file (str): Path to the file.
        Returns:
            None
        """
        self.__convert(file=file)
        self.__drop_columns(file=file)
        self.__scale()
        if not self.training:
            self.__gate()

    def __aggregate(self) -> None:
        """
        Aggregates individual dataframes into one big dataframe, which is used for the model training.
        Returns:
            None
        """
        if self.aggregated is not None:
            self.aggregated = pd.concat([self.aggregated, self.data])
        else:
            self.aggregated = self.data

    def get_aggregated(self) -> np.ndarray:
        """
        Returns:
            np.array: Returns aggregated dataset containing all data points without labels.
        """
        self.__check_extension()
        for file in self.files_list:
            self.__run_transformation(file)
            self.__add_labels(file)
            self.__aggregate()
        return np.array(self.aggregated)

    def get_data(self, file: str) -> np.ndarray:
        """
        Returns:
            np.array: Returns transformed dataset corresponding to the individual input file.
        """
        self.__run_transformation(file)
        return self.data

    def get_labels(self) -> np.ndarray:
        """
        Returns:
            np.array: Returns vector containing strings with the label of data points. Corresponds to the return of the
                get_dataframe() method.
        """
        return np.array(self.labels)

    def get_features(self) -> list:
        """
        Returns:
            dict: Returns dictionary containing features for each file.
        """
        unique = [list(x) for x in set(tuple(x) for x in self.features.values())]
        if len(unique) == 1:
            return unique[0]
        else:
            raise ValueError("Files are from different FCs")
            # TODO: Raise issues because files are coming from different sources


class DataPreparation:
    """
    DataPreparation class is used to create training, validation and test sets for the model training in the TensorFlow
    Dataset format.
    """

    def __init__(self, dataframe: np.ndarray, labels: np.ndarray = None, batch_size: int = 256) -> None:
        """
        Args:
            dataframe (np.ndarray): Dataframe containing data points for the training.
            labels (np.ndarray): Vector of strings containing labels for the data points corresponding to the dataframe.
            batch_size (int): Size of the batches for training.
        """
        self.dataframe = dataframe
        self.labels = labels
        self.labels_ints = None
        self.batch_size = batch_size

    def __map_labels(self) -> None:
        """
        Maps labels, which are strings, to the numerical values used in the dataset.
        Returns:
            None
        """
        self.labels, self.labels_ints = np.unique(self.labels, return_inverse=True)
        self.labels_ints = tf.convert_to_tensor(self.labels_ints, dtype=tf.float32)

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Creates TF Datasets from given dataframe, UMAP embeddings and labels, which are converted into the ints
        according to the conversion map.
        Returns:
            (tf.data.Dataset, tf.data.Dataset): Tuple of tf.data.Datasets for training and validation.
        """
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
        """
        Returns:
            dict: Dictionary containing mapping from labels to ints.
        """
        return {index: value for index, value in enumerate(self.labels)}
