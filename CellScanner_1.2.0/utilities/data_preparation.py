from typing import Tuple

import tensorflow as tf
import pandas as pd
import numpy as np
import fcsparser

from utilities.settings import Settings


class FilePreparation:
    """
    FilePreparation class is used for the importing and transforming files for the analysis. The main format is .fcs.
    Other supported formats are csv and tsv. Files can be loaded in batch by specifying the files arguments when
    initializing class and later calling get_aggregated() method or processed individually with the get_data() method.
    """

    def __init__(self, files: list, settings: Settings, col_to_drop: list = None) -> None:
        """
        Args:
            files (list, optional): List of strings containing absolute path to the desired files.
            col_to_drop (list, optional): Optional list of strings containing file columns that will be excluded.
                Default columns are "Width" and "Time".
        """
        # TODO: Columns to drop from config when implemented
        self.files_list = files
        self.fc_type = settings.fc_type
        self.gating_type = settings.gating_type
        self.col_names = col_to_drop    # TODO: Implement in Settings
        self.data = None
        self.aggregated = None
        self.labels = []

    def __check_extension(self):
        # TODO: Consider showing warning message
        """
        Checks if the provided files have correct extension, which can be .fcs, .csv or .tsv.
        Returns:
            None
        """
        for file in self.files_list:
            if file.split(".")[-1] not in ["fcs", "csv", "tsv"]:
                self.files_list.remove(file)

    def __convert(self, file: str) -> None:
        """
        Converts given file into pandas dataframe.
        Args:
            file (str): path to the file.
        """
        meta, self.data = fcsparser.parse(file, meta_data_only=False, reformat_meta=False)

    def __drop_columns(self) -> None:
        # TODO: Add extra columns from config
        """
        By default, drops only "time" column. Alternatively, extra columns can be specified in the settings.
        Returns:
            None
        """
        if self.col_names is None:
            self.col_names = ["Time", "Width"]  # TODO: Remove Width (its only used for classifier_14_512.h5)
        try:
            self.data.drop(self.col_names, axis=1, inplace=True)
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
        if self.fc_type == "Accuri":
            for index, row in self.data.iterrows():
                if (10 ** row["FL3-A"]) > (0.0241 * (10 ** row["FL3-A"]) ** 1.0996):
                    self.data.drop(index)
                elif row["FSC-A"] > 5.0 and row["SSC-A"] > 4.0:     # Values are after log10 transformation
                    self.data.drop(index)
                elif row["FSC-A"] > (row["FSC-H"] + 0.5) or row["FSC-A"] < (row["FSC-H"] - 0.5):
                    self.data.drop(index)
        elif self.fc_type == "CytoFlex":
            for index, row in self.data.iterrows():
                if row["FL3-A"] > (1.5 * row["'FL1-A'"] - 2.8) or row["FL2-A"] > (2.5 * row["FL1-A"] - 9):
                    self.data.drop(index)
                elif row["FSC-A"] > (row["FSC-H"] + 0.6) or row["FSC-A"] > (row["FSC-H"] - 0.6):
                    self.data.drop(index)
        # TODO: Do classifier to detect blanks and throw them away from dataset. Delete the thing above.

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

    def __add_labels(self, file: str) -> None:
        """
        Extracts labels from the file name (assuming they are named correctly) and adds to the labels array, which is
        used during training for creating label maps.
        Args:
            file (str): path to the file.
        Returns:
            None
        """
        # TODO: Consider regex for finding bacteria name
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
            file (str): path to the file.
        Returns:
            None
        """
        self.__convert(file)
        self.__drop_columns()
        self.__scale()
        self.__gate()

    def get_data(self, file: str) -> np.ndarray:
        """
        Returns:
            np.array: Returns transformed dataset corresponding to the individual input file.
        """
        self.__run_transformation(file)
        return self.data

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

    def get_labels(self) -> np.ndarray:
        """
        Returns:
            np.array: Returns vector containing strings with the label of data points. Corresponds to the return of the
                get_dataframe() method.
        """
        # TODO: Works good in diagnostics as long as there is a single file provided.
        # Solution, instead of using just ndarray, you dict with file_name: ndarray of labels pair
        return np.array(self.labels)

    def get_labels_shape(self) -> tuple:
        """
        Returns:
            tuple: Tuple containing value of unique labels.
        """
        return np.array(list(set(self.labels))).shape


class DataPreparation:
    """
    DataPreparation class is used to create training, validation and test sets for the model training in the TensorFlow
    Dataset format.
    """
    def __init__(self, dataframe: np.ndarray, embeddings: np.ndarray, labels: np.ndarray, batch_size: int) -> None:
        """
        Args:
            dataframe (np.ndarray): Dataframe containing data points for the training.
            labels (np.ndarray): Vector of strings containing labels for the data points corresponding to the dataframe.
            batch_size (int): Size of the batches for training.
        """
        self.dataframe = dataframe
        self.embeddings = embeddings
        self.labels = labels
        self.labels_ints = None
        self.batch_size = batch_size
        self.__map_labels()

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
        dataset = tf.data.Dataset.from_tensor_slices(({"standard": self.dataframe, "embeddings": self.embeddings},
                                                      self.labels_ints))
        dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset).numpy())    # TODO: Check if this is correct
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
