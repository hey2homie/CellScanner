import argparse
import os

from utilities.settings import Settings, SettingsOptions, ModelsInfo
from utilities.classification_utils import ClassificationResults, ClassificationTraining


def run_prediction():
    if arguments.path is None or check_path(arguments.path) is None:
        raise argparse.ArgumentTypeError("Incorrect path or empty directory")
    files = os.listdir(arguments.path)
    if arguments.command == "prediction":
        ClassificationResults(files=files, settings=settings, models_info=models_info)
        pass
    elif arguments.command == "diagnostics":
        classifier = ClassificationResults(files=files, settings=settings, models_info=models_info, diagnostics=True)
        classifier.run_diagnostics()
        pass
    else:
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
                if training_type in ["Autoencoder", "Classifier"]:
                    break
        if model_name and training_type:
            # TODO: Implement choice of training type
            # training = ClassificationTraining(training_type=training_type,files=files, model_name=model_name,
            #                                  settings=settings, models_info=models_info)
            # training.run_training()
            pass
        else:
            raise argparse.ArgumentTypeError("No arguments provided")


def run_settings():
    if arguments.show:
        print(settings.attributes)  # TODO: Nicely format the output
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
        except AttributeError:  # If attribute is not in SettingsOptions
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
                        choices=["prediction", "training", "settings", "diagnostics"])

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

    if arguments.command == "prediction" or arguments.command == "training" or arguments.command == "diagnostics":
        run_prediction()
    elif arguments.command == "settings":
        run_settings()


if __name__ == "__main__":
    settings = Settings()
    models_info = ModelsInfo()
    main()
