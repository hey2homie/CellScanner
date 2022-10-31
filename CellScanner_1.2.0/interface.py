import argparse
import os
import yaml

from utilities.settings import Settings, SettingsOptions, ModelsInfo
from utilities.classification_utils import ClassificationModel, AutoEncoder


def run_prediction():
    if arguments.path is None or check_path(arguments.path) is None:
        raise argparse.ArgumentTypeError("Incorrect path or empty directory")
    files = [os.path.join(arguments.path, file) for file in os.listdir(arguments.path)]
    if arguments.command == "predict":
        model = ClassificationModel(settings=settings, model_info=models_info, files=files)
        model.run_classification()
    elif arguments.command == "validate":
        model = ClassificationModel(settings=settings, model_info=models_info, files=files)
        model.run_diagnostics()
    elif arguments.command == "train":
        if arguments.name:
            model_name = arguments.name[0]
        else:
            while True:
                model_name = input("Enter model name: ")
                if model_name:
                    break
        if arguments.model:
            training_type = arguments.model[0]
        else:
            while True:
                training_type = input("Enter model type: ")
                if training_type in ["autoencoder", "classifier"]:
                    break
        if model_name and training_type:
            if training_type == "classifier":
                model = ClassificationModel(settings=settings, model_info=models_info, files=files)
                model.run_training(name=model_name)
            else:
                autoencoder = AutoEncoder(settings=settings, model_info=models_info, files=files)
                autoencoder.run_training(name=model_name)
        else:
            raise argparse.ArgumentTypeError("No arguments provided")


def run_settings():
    if arguments.show:
        print(yaml.dump(vars(settings), default_flow_style=False))
    elif arguments.change:
        try:
            available_values = getattr(SettingsOptions, arguments.change[0]).value
            while True:
                value = input(f"Available options are {available_values}: ")
                if value in available_values:
                    setattr(settings, arguments.change[0], value)
                    break
                else:
                    print("Invalid option")
        except AttributeError:
            while True:
                value = input(f"Enter new value for {arguments.change[0]}: ")
                if value:
                    setattr(settings, arguments.change[0], int(value))
                    break
                else:
                    print("Invalid option")
        settings.save_settings(command_line=True)
    elif arguments.change is None or arguments.show is None:
        raise argparse.ArgumentTypeError("No arguments provided")


def check_path(path):
    if not os.path.exists(path):
        return None
    elif not os.listdir(path):
        return None
    else:
        return path


def main():
    parser = argparse.ArgumentParser(description="CellScanner")

    parser.add_argument("command", type=str, help="Command to run",
                        choices=["predict", "train", "validate", "settings"])

    # General
    parser.add_argument("-p", "--path", type=str, help="Path to files to process")

    # Settings
    parser.add_argument("-s", "--show", dest="show", action="store_true", help="Show settings")
    parser.add_argument("-c", "--change", type=str, help="Change settings",
                        choices=["fc_type", "model", "results", "vis_type", "num_umap_cores", "hardware", "num_batches",
                                 "num_epochs", "lr", "lr_scheduler", "lr_reduced", "gating_type", "autoencoder",
                                 "mse_threshold"],    # TODO: Parse attributes from the Settings class
                        nargs=1)

    # Retraining
    parser.add_argument("-m", "--model", type=str, help="What type of model to train",
                        choices=["autoencoder", "classifier"], nargs=1)
    parser.add_argument("-n", "--name", type=str, help="Name of the model", nargs=1)

    global arguments
    arguments = parser.parse_args()

    if arguments.command == "predict" or arguments.command == "train" or arguments.command == "diagnose":
        run_prediction()
    elif arguments.command == "settings":
        run_settings()


if __name__ == "__main__":
    settings = Settings()
    models_info = ModelsInfo()
    main()
