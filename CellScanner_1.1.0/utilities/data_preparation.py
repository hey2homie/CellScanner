import pandas as pd
import numpy as np
import fcsparser

# TODO: Extend Documentation
"""
DataPreparation class takes a string containing location of the files and sequentially performs the following 
operations: 1) conversion of files in vcf format to csv format; 2) drops unnecessary columns; 3) logarithmic 
transformation; 4) aggregation into single dataframe.
"""


class DataPreparation:

    def __init__(self, col_to_drop: list = None) -> None:
        self.files_list = []
        self.col_names = col_to_drop
        self.data = None
        self.aggregated = None

    def add_files(self, file: str) -> None:
        self.files_list.append(file)

    def __convert(self, file: str) -> None:
        if file.split(".")[-1] == "fcs":  # TODO: Add extra formats just in case
            meta, self.data = fcsparser.parse(file, meta_data_only=False, reformat_meta=False)
        self.data["Species"] = file.split("/")[-1].split(".")[0]

    def __drop_columns(self, col_names: list = None) -> None:
        if self.col_names is None:
            col_names = ["Width", "Time"]  # TODO: Width column?
        self.data.drop(col_names, axis=1, inplace=True)

    def __scale(self) -> None:
        for column in self.data.select_dtypes(include=[np.number]).columns:
            with np.errstate(divide="ignore"):
                self.data[column] = np.log10(self.data[column].values)
        self.data.replace([np.inf, -np.inf], 0, inplace=True)

    def __aggregate(self) -> None:
        if self.aggregated is not None:
            self.aggregated = pd.concat([self.aggregated, self.data])
        else:
            self.aggregated = self.data

    def get_object(self) -> pd.DataFrame:
        for file in self.files_list:
            self.__convert(file)
            self.__drop_columns()
            self.__scale()
            self.__aggregate()
        return self.aggregated
