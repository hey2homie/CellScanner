import yaml
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
    gating_type = ["Autoencoder", "Machine"]

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
    cols_to_drop_accuri = None
    cols_to_drop_cytoflex = None
    mse_threshold = None

    def __init__(self) -> None:
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
                yaml.dump(attributes, config)
            else:
                yaml.dump(vars(self), config)


class ModelsInfo:
    classifiers = None
    autoencoders = None
    classifier_name = None
    autoencoder_name = None

    def __init__(self) -> None:
        self.__load_settings()
        self.__check_models()

    def __load_settings(self) -> None:
        with open(".configs/models_info.yml", "r") as config:
            config_content = yaml.load(config, Loader=yaml.SafeLoader)
            self.classifiers = config_content["classifiers"]
            self.autoencoders = config_content["autoencoders"]

    def __check_models(self) -> None:
        classifiers = os.listdir("./classifiers/")
        autoencoders = [encoder for encoder in os.listdir("./autoencoders/") if encoder.endswith(".h5")]
        self.classifiers = {model: self.classifiers[model] for model in classifiers}
        self.autoencoders = {model: self.autoencoders[model] for model in autoencoders}
        self.save_info()
        # TODO: Remove .npy files from .clean_data if ae is removed

    def get_features(self) -> list:
        return self.classifiers[self.classifier_name][1]["features"]

    def get_features_shape_ae(self) -> int:
        return int(self.autoencoders[self.autoencoder_name][2]["num_features"])

    def get_labels_map(self) -> dict:
        return self.classifiers[self.classifier_name][1]["labels_map"]

    def get_features_shape_classifier(self) -> int:
        return int(self.classifiers[self.classifier_name][0]["features_shape"])

    def get_labels_shape(self) -> int:
        return int(self.classifiers[self.classifier_name][2]["labels_shape"])

    def get_readable(self, model: str, name: str) -> str:
        if model == "classifier":
            labels_map = self.classifiers[name][1]["labels_map"]    # TODO: Use method get_labels_map
            labels = [v for _, v in labels_map.items()]
            return ", ".join(labels)
        else:
            features = self.autoencoders[name][1]["features"]
            return ", ".join(features)

    def add_classifier(self, name: str, labels_map: dict, features_shape: int, labels_shape: int) -> None:
        labels_map = {key: str(value) for key, value in labels_map.items()}
        self.classifier_name = name
        self.classifiers[name] = [{"features_shape": features_shape}, {"labels_map": labels_map},
                                  {"labels_shape": labels_shape}]

    def add_autoencoder(self, name: str, fc_type: str, features: list, num_features: int) -> None:
        self.autoencoder_name = name
        self.autoencoders[name] = [{"fc_type": fc_type}, {"features": features}, {"num_features": num_features}]

    def save_info(self) -> None:
        with open(".configs/models_info.yml", "w") as config:
            yaml.dump({"autoencoders": self.autoencoders, "classifiers": self.classifiers}, config,
                      default_flow_style=False)
