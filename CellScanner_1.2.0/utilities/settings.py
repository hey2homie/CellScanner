import ast
import yaml
import inspect
import os
from glob import glob
from enum import Enum


class SettingsOptions(Enum):
    """
    Enum containing options for the Combo Boxes used in SettingsWindow.
    """
    fc_type = ["Accuri", "Cytoflex"]
    model = [os.path.basename(file) for file in glob("./classifiers/*.h5")]
    autoencoder = [os.path.basename(file) for file in glob("./autoencoders/*.h5")]
    vis_type = ["UMAP", "Channels"]
    vis_dims = ["2", "3"]
    vis_channels_accuri = ["FL1-A", "FL2-A", "FL3-A", "FL4-A", "FSC-H", "SSC-H", "FL1-H", "FL2-H", "FL3-H", "FL4-H",
                           "Width", "Time"]
    vis_channels_cytoflex = ["FSC-H", "FSC-A", "SSC-H", "SSC-A", "FL1-H", "FL1-A", "FL4-H", "FL4-A", "FL3-red-H",
                             "FL3-red-A", "APC-A750-H", "APC-A750-A", "VSSC-H", "VSSC-A", "KO525-H", "KO525-A",
                             "FL2-orange-H", "FL2-orange-A", "mCherry-H", "mCherry-A", "PI-H", "PI-A", "FSC-Width",
                             "Time"]
    hardware = ["GPU", "CPU"]
    lr = ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"]
    lr_scheduler = ["Constant", "Time Decay", "Step Decay", "Exponential Decay"]
    gating_type = ["Line", "Autoencoder", "Machine"]

    @classmethod
    def get_dictionary(cls) -> dict:
        return {key.name: key.value for key in cls}


class Settings:
    """
    Attributes of this class are used throughout the project to determine settings for various operations. Values of
    attributes can be modified either in the setting window or manually by changing entries in the configuration file.
    """
    fc_type = None
    hardware = None
    results = None
    vis_type = None
    vis_dims = None
    vis_channels_accuri = None
    vis_channels_cytoflex = None
    num_umap_cores = None
    model = None
    num_batches = None
    num_epochs = None
    lr_scheduler = None
    lr = None
    gating_type = None
    autoencoder = None
    mse_threshold = None

    def __init__(self) -> None:
        self.__load_settings()
        self.attributes = inspect.getmembers(self, lambda x: not (inspect.isroutine(x)))
        self.attributes = dict([a for a in self.attributes if not (a[0].startswith('__') and a[0].endswith('__'))])

    def __load_settings(self) -> None:
        """
        Loads configuration file, from which class attributes receive their values.
        Returns:
            None
        """
        with open(".configs/config.yml", "r") as config:
            settings = yaml.load(config, Loader=yaml.SafeLoader)
            for key, value in settings.items():
                setattr(self, key, value)

    def save_settings(self, command_line: bool = False) -> None:
        """
        Save settings by rewriting configuration file.
        Returns:
            None
        """
        with open(".configs/config.yml", "w") as config:
            if command_line:
                attributes = vars(self)
                attributes.pop("attributes")
                yaml.dump(attributes, config)
            else:
                yaml.dump(self.attributes, config)

    def get_attributes(self) -> dict:
        """
        Returns:
            attributes (dict): a dictionary of attributes and their values.
        """
        return self.attributes

    def set_attributes(self, attributes: dict) -> None:
        """
        Args:
            attributes (dict): a dictionary of attributes and their values.
        """
        self.attributes = attributes
        for key, value in self.attributes.items():
            setattr(self, key, value)


class ModelsInfo:
    """
    Attributes of this class are used throughout the project to determine settings for operations involving usage of NN.
    Values of attributes in config cannot be modified but added with each new model.
    """
    # TODO: Add attribute for autoencoder model
    # TODO: Add methods to save/load autoencoder
    classifiers = None
    autoencoders = None
    classifier_name = None
    autoencoder_name = None

    def __init__(self) -> None:
        self.__load_settings()
        self.__check_models()

    def __load_settings(self) -> None:
        """
        Loads configuration file, from which class attribute receive their values.
        """
        with open(".configs/models_info.yml", "r") as config:
            config_content = yaml.load(config, Loader=yaml.SafeLoader)
            self.classifiers = config_content["classifiers"]
            self.autoencoders = config_content["autoencoders"]

    def __check_models(self) -> None:
        """
        Checks if configuration file is up-to-date with the available classifiers. If classifier is no longer present in
        folder, removes it from configuration file
        Returns:
            None
        """
        classifiers = os.listdir("./classifiers/")
        autoencoders = [encoder for encoder in os.listdir("./autoencoders/") if encoder.endswith(".h5")]
        self.classifiers = {model: self.classifiers[model] for model in classifiers}
        self.autoencoders = {model: self.autoencoders[model] for model in autoencoders}
        self.save_info()

    def set_autoencoder_name(self, name: str) -> None:
        """
        Args:
            name (str): name of the model.
        Returns: None
        """
        self.autoencoder_name = name

    def get_features_shape_ae(self) -> list:
        """
        Returns:
            features (list): a list of features.
        """
        return self.autoencoders[self.autoencoder_name][2]["num_features"]

    def set_classifier(self, name: str) -> None:
        """
        Args:
            name (str): name of the model.
        Returns: None
        """
        self.classifier_name = name

    def get_labels_map(self) -> dict:
        """
        Returns:
            labels_map (dict): a dictionary of classifiers and their labels.
        """
        return self.classifiers[self.classifier_name][0]["labels_map"]

    def get_features_shape_classifier(self) -> tuple:
        """
        Returns:
            features_shape (tuple): a tuple of the shape of the features.
        """
        return ast.literal_eval(self.classifiers[self.classifier_name][1]["features_shape"])

    def get_labels_shape(self) -> tuple:
        """
        Returns:
            labels_shape (tuple): a tuple of the shape of the labels.
        """
        return ast.literal_eval(self.classifiers[self.classifier_name][2]["labels_shape"])

    def add_classifier(self, name: str, labels_map: dict, features_shape: tuple, labels_shape: tuple) -> None:
        """
        Adds information about new model to the classifiers class attributes, which is later being saved to the
        configuration files, and saves models information to configuration file.
        Args:
            name (str): name of the model.
            labels_map (dict): a dictionary with pair int:str, where int is the class predicted by model and str name of
            the bacteria.
            features_shape (tuple): a tuple of the shape of the features.
            labels_shape (tuple): a tuple of the shape of the labels.
        Returns:
            None
        """
        labels_map = {key: str(value) for key, value in labels_map.items()}
        self.classifiers[name] = [{"labels_map": labels_map}, {"features_shape": str(features_shape)},
                                  {"labels_shape": str(labels_shape)}]

    def add_autoencoder(self, name: str, fc_type: str, features: list, num_features: int) -> None:
        self.autoencoders[name] = [{"fc_type": fc_type}, {"features": features}, {"num_features": num_features}]

    def save_info(self) -> None:
        """
        Saves models information to configuration file.
        Returns: None
        """
        with open(".configs/models_info.yml", "w") as config:
            yaml.dump({"autoencoders": self.autoencoders, "classifiers": self.classifiers}, config,
                      default_flow_style=False)
