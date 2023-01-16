import yaml
import os
from glob import glob
from enum import Enum

from .helpers import set_tf_hardware


class SettingsOptions(Enum):
    """
    Enum containing options for the Combo Boxes used in SettingsWindow.
    """
    fc_type = ["Accuri", "Cytoflex"]
    model = [os.path.basename(file) for file in glob("./classifiers/*.h5")]
    autoencoder = [os.path.basename(file) for file in glob("./autoencoders/*.h5")]
    vis_type = ["UMAP", "Channels"]
    vis_dims = ["2", "3"]
    # TODO: Remove vis_channels and rely solely on the channels of the autoencoder
    vis_channels_accuri = ["FSC-A", "SSC-A", "FL1-A", "FL2-A", "FL3-A", "FL4-A", "FSC-H", "SSC-H", "FL1-H", "FL2-H",
                           "FL3-H", "FL4-H", "Width"]
    vis_channels_cytoflex = ["FSC-H", "FSC-A", "SSC-H", "SSC-A", "FL1-H", "FL1-A", "FL4-H", "FL4-A", "FL3-red-H",
                             "FL3-red-A", "APC-A750-H", "APC-A750-A", "VSSC-H", "VSSC-A", "KO525-H", "KO525-A",
                             "FL2-orange-H", "FL2-orange-A", "mCherry-H", "mCherry-A", "PI-H", "PI-A", "FSC-Width"]
    cols_to_drop_accuri = ["FL1-A", "FL2-A", "FL3-A", "FL4-A", "FSC-H", "SSC-H", "FL1-H", "FL2-H", "FL3-H", "FL4-H",
                           "Width", "Time"]
    cols_to_drop_cytoflex = ["FSC-H", "FSC-A", "SSC-H", "SSC-A", "FL1-H", "FL1-A", "FL4-H", "FL4-A", "FL3-red-H",
                             "FL3-red-A", "APC-A750-H", "APC-A750-A", "VSSC-H", "VSSC-A", "KO525-H", "KO525-A",
                             "FL2-orange-H", "FL2-orange-A", "mCherry-H", "mCherry-A", "PI-H", "PI-A", "FSC-Width",
                             "Time"]
    hardware = ["GPU", "CPU"]
    lr = ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"]
    lr_scheduler = ["Constant", "Time Decay", "Step Decay", "Exponential Decay"]
    gating_type = ["Autoencoder", "Machine"]

    @classmethod
    def get_dictionary(cls) -> dict:
        """
        Returns a dictionary containing the Enum values.
        Returns:
            dict: Dictionary containing the Enum values.
        """
        return {key.name: key.value for key in cls}


class Settings:
    """
    Attributes of this class are used throughout the project to determine settings for various operations. Values of
    attributes can be modified either in the setting window or manually by changing entries in the configuration file.

    Attributes:
    ----------
    fc_type: str
        Flow cytometer type. Either Accuri or CytoFlex.
    hardware: str
        Hardware to use for training. Either CPU or GPU.
    results: str
        Path to results directory.
    vis_type: str
        Type of visualization to use. Either UMAP or Channels.
    vis_dims: int
        Number of dimensions to use for visualization. Either 2 or 3.
    vis_channels_accuri: list
        Channels to use for visualization when using Accuri flow cytometer.
    vis_channels_cytoflex: list
        Channels to use for visualization when using CytoFlex flow cytometer.
    num_umap_cores: str
        Number of cores to use for UMAP.
    model: str
        Name of the model to be used for classification.
    num_batches: str
        Number of batches for creating training dataset.
    num_epochs: str
        Number of epochs for training.
    lr_scheduler: str
        Type of learning rate scheduler to use. Either Constant, Time Decay, Step Decay, or Exponential Decay.
    lr: str
        Learning rate.
    gating_type: str
        Type of gating to use. Either Autoencoder or Machine.
    autoencoder: str
        Name of the autoencoder to be used for gating.
    cols_to_drop_accuri: list
        Columns to drop when using Accuri flow cytometer.
    cols_to_drop_cytoflex: list
        Columns to drop when using CytoFlex flow cytometer.
    mse_threshold: str
        Threshold for autoencoder gating. Can be adjusted after making predictions.
    mlp: bool
        Indicates usage of NN model from CellScanner V1.
    number_of_clusters: int
        Number of clusters to use in data preparation for autoencoder.
    blank_threshold: int
        Clusters containing more than this number of blank events will be removed.
    softmax_prob_threshold: float
        Threshold for softmax probability. Events with probability less than this value are labelled as low probability
        predictions.
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
    cols_to_drop_accuri = None
    cols_to_drop_cytoflex = None
    mse_threshold = None
    mlp = None
    number_of_clusters = None
    blank_threshold = None
    softmax_prob_threshold = None

    def __init__(self) -> None:
        """
        Initializes the Settings class by loading the configuration file and setting the attribute's values. Also, sets
        tensorflow device for the current session
        """
        with open(".configs/config.yml", "r") as config:
            settings = yaml.load(config, Loader=yaml.SafeLoader)
            for key, value in settings.items():
                setattr(self, key, value)
        set_tf_hardware(self.hardware)
        if not os.path.isdir(self.results):
            os.mkdir(self.results)

    def save_settings(self, command_line: bool = False) -> None:
        """
        Save settings by rewriting configuration file.
        Args:
            command_line (bool, optional): Whether or not the process is run from the command line. Defaults to False.
        Returns:
            None.
        """
        with open(".configs/config.yml", "w") as config:
            if command_line:
                attributes = vars(self)
                yaml.dump(attributes, config)
            else:
                yaml.dump(vars(self), config)


class ModelsInfo:
    """
    This object contains metadata about the models that are available for prediction, validation. Used throughout the
    project.

    Attributes:
    ----------
    classifiers: dict
        Dictionary containing metadata about the classifiers.
    autoencoders: dict
        Dictionary containing metadata about the autoencoders.
    classifier_names: str
        Current classifier name.
    autoencoder_name: str
        Current autoencoder name.
    """
    classifiers = None
    autoencoders = None
    classifier_name = None
    autoencoder_name = None

    def __init__(self) -> None:
        """
        Initializes the ModelsInfo class by loading the models metadata file and setting the attribute's values.
        Returns:
            None.
        See Also:
            :meth:`__load_settings`.
            :meth:`__check_models`.
        """
        self.__load_settings()
        self.__check_models()

    def __load_settings(self) -> None:
        """
        Loads the models metadata file and sets the attribute's values.
        Returns:
            None.
        """
        with open(".configs/models_info.yml", "r") as config:
            config_content = yaml.load(config, Loader=yaml.SafeLoader)
            self.classifiers = config_content["classifiers"]
            self.autoencoders = config_content["autoencoders"]

    def __check_models(self) -> None:
        """
        Checks if the models specified in the configuration file are available. If not, they are deleted from the
        configuration file.
        Returns:
            None.
        See Also:
            :meth:`save_info`.
        """
        classifiers = [os.path.basename(file) for file in glob("./classifiers/*.h5")]
        autoencoders = [os.path.basename(file) for file in glob("./autoencoders/*.h5")]
        self.classifiers = {model: self.classifiers[model] for model in classifiers}
        self.autoencoders = {model: self.autoencoders[model] for model in autoencoders}
        self.save_info()

    def get_features_ae(self) -> list:
        """
        Returns:
            list: List of features used for autoencoder training.
        """
        return self.autoencoders[self.autoencoder_name][1]["features"]

    def get_features_shape_ae(self) -> int:
        """
        Returns:
            int: Number of features used for autoencoder training.
        """
        return int(self.autoencoders[self.autoencoder_name][2]["num_features"])

    def get_labels_map(self) -> dict:
        """
        Returns:
            dict: Dictionary containing the mapping between the labels and their corresponding integer values.
        """
        return self.classifiers[self.classifier_name][2]["labels_map"]

    def get_features_shape_classifier(self) -> int:
        """
        Returns:
            int: Number of features used for classifier training.
        """
        return int(self.classifiers[self.classifier_name][1]["features_shape"])

    def get_labels_shape(self) -> int:
        """
        Returns:
            int: Number of labels used for classifier training.
        """
        return int(self.classifiers[self.classifier_name][3]["labels_shape"])

    def get_readable(self, model: str, name: str) -> str:
        """
        Args:
            model (str): Type of model. Either classifier or autoencoder.
            name (str): Name of the model.
        Returns:
            str: Readable list of features or labels.
        """
        if model == "classifier" or model == "model":
            labels_map = self.classifiers[name][2]["labels_map"]
            labels = [v for _, v in labels_map.items()]
            return ", ".join(labels)
        else:
            features = self.autoencoders[name][1]["features"]
            return ", ".join(features)

    def add_classifier(self, name: str, fc_type: str, labels_map: dict, features_shape: int, labels_shape: int,
                       files_used: list, autoencoder_used: str) -> None:
        """
        Adds a classifier to the configuration file.
        Args:
            name (str): Name of the classifier.
            fc_type (str): Type of the flow cytometry.
            labels_map (dict): Dictionary containing the mapping between the labels and their corresponding integer
                values.
            features_shape (int): Number of features used for classifier training.
            labels_shape (int): Number of labels used for classifier training.
            files_used (list): List of files used for classifier training.
            autoencoder_used (str): Name of the autoencoder used for classifier training.
        Returns:
            None.
        """
        labels_map = {key: str(value) for key, value in labels_map.items()}
        files_used = [file.split("/")[-1] for file in files_used]
        self.classifier_name = name
        self.classifiers[name] = [{"fc_type": fc_type}, {"features_shape": features_shape}, {"labels_map": labels_map},
                                  {"labels_shape": labels_shape}, {"files_used": files_used},
                                  {"autoencoder": autoencoder_used}]

    def add_autoencoder(self, name: str, fc_type: str, features: list, num_features: int) -> None:
        """
        Adds an autoencoder to the configuration file.
        Args:
            name (str): Name of the autoencoder.
            fc_type (str): Type of the flow cytometry.
            features (list): List of features used for autoencoder training.
            num_features (int): Number of features used for autoencoder training.
        Returns:
            None.
        """
        self.autoencoder_name = name
        self.autoencoders[name] = [{"fc_type": fc_type}, {"features": features}, {"num_features": num_features}]

    def save_info(self) -> None:
        """
        Saves current metadata to the configuration file.
        Returns:
            None.
        """
        with open(".configs/models_info.yml", "w") as config:
            yaml.dump({"autoencoders": self.autoencoders, "classifiers": self.classifiers}, config,
                      default_flow_style=False, sort_keys=False)
