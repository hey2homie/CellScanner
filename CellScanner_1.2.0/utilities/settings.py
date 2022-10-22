import ast
import yaml
import inspect
import os


# TODO: Rename file


class Settings:
    """
    Attributes of this class are used throughout the project to determine settings for various operations. Values of
    attributes can be modified either in the setting window or manually by changing entries in the configuration file.
    """
    # TODO: Add docstrings
    # TODO: Consider using slots for the class attributes
    # TODO: Add attribute for autoencoder model
    fc_type = None
    model = None
    results = None
    vis_type = None
    vis_dims = None
    vis_channels_accuri = None
    vis_channels_cytoflex = None
    num_umap_cores = None
    hardware = None
    lr = None
    lr_scheduler = None
    lr_reduced = None
    num_batches = None
    num_epochs = None
    cols_to_drop_accuri = None
    cols_to_drop_cytoflex = None
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

    def save_settings(self) -> None:
        """
        Save settings by rewriting configuration file.
        Returns:
            None
        """
        with open(".configs/config.yml", "w") as config:
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

    def get_features(self) -> list:
        """
        Returns:
            features (list): a list of features.
        """
        return self.classifiers[self.classifier_name]["features"]

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

    def get_features_shape(self) -> tuple:
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

    def add_model(self, name: str, labels_map: dict, features_shape: tuple, labels_shape: tuple) -> None:
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

    def save_info(self) -> None:
        """
        Saves models information to configuration file.
        Returns: None
        """
        with open(".configs/models_info.yml", "w") as config:
            yaml.dump({"autoencoders": self.autoencoders, "classifiers": self.classifiers}, config,
                      default_flow_style=False)