import argparse
import os
import yaml

from utilities.settings import Settings, SettingsOptions, ModelsInfo
from utilities.classification_utils import ClassificationModel, AutoEncoder
from utilities.visualizations import MplVisualization
from utilities.helpers import create_output_dir, save_cell_counts


def run_prediction() -> None:
    """
    Depending on the command, runs the prediction, training or validation.
    Returns:
        None.
    Raises:
        argparse.ArgumentTypeError: In case of folder/file not existing.
        argparse.ArgumentTypeError: If incorrect arguments are provided.
    """
    if arguments.path is None or check_path(arguments.path) is None:
        raise argparse.ArgumentTypeError("Incorrect path or empty directory")
    if os.path.isdir(arguments.path):
        files = [os.path.join(arguments.path, file) for file in os.listdir(arguments.path)]
    else:
        files = [arguments.path]
    if arguments.command == "predict":
        model = ClassificationModel(settings=settings, model_info=models_info, files=files, model_type="classifier",
                                    name=settings.model)
        results = model.run_classification()
        output_dir = create_output_dir(path=settings.results)
        visualizations = MplVisualization(output_path=output_dir)
        visualizations.save_predictions_visualizations(inputs=results, settings=settings)
        save_cell_counts(path=output_dir, inputs=results, mse_threshold=settings.mse_threshold)
        print("\nClassification is finished")
    elif arguments.command == "validate":
        model = ClassificationModel(settings=settings, model_info=models_info, files=files, model_type="classifier",
                                    name=settings.model)
        model.run_diagnostics()
        print("\nValidation is finished")
    elif arguments.command == "train":
        if arguments.name:
            model_name = arguments.name[0]
        else:
            while True:
                model_name = input("Enter model name: ")
                if model_name:
                    model_name = model_name
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
                model_name += ".h5"
                model = ClassificationModel(settings=settings, model_info=models_info, files=files,
                                            model_type="classifier", name=model_name, training_cls=True)

            else:
                model = AutoEncoder(settings=settings, model_info=models_info, files=files, model_type="ae",
                                    name=model_name)
            model.run_training()
            print("\nTraining is finished")
        else:
            raise argparse.ArgumentTypeError("No arguments provided")


def run_settings() -> None:
    """
    Shows settings or allows to change them.
    Returns:
        None.
    Raises:
        argparse.ArgumentTypeError: If incorrect arguments are provided.
    """
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


def check_path(path: str) -> None or str:
    """
    Checks if the path is valid.
    Args:
        path (str): Path to check.
    Returns:
        None or str: None if path is invalid, path if valid.
    """
    if not os.path.exists(path):
        return None
    else:
        return path


def main() -> None:
    """
    Main function. Initializes argparse object and sets possible commands.
    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="CellScanner")
    parser.add_argument("command", type=str, help="Command to run",
                        choices=["predict", "train", "validate", "settings"])
    parser.add_argument("-p", "--path", type=str, help="Path to files to process")
    parser.add_argument("-s", "--show", dest="show", action="store_true", help="Show settings")
    parser.add_argument("-c", "--change", type=str, help="Change settings",
                        choices=list(vars(settings).keys()), nargs=1)
    parser.add_argument("-m", "--model", type=str, help="What type of model to train",
                        choices=["autoencoder", "classifier"], nargs=1)
    parser.add_argument("-n", "--name", type=str, help="Name of the model", nargs=1)
    global arguments
    arguments = parser.parse_args()
    if arguments.command == "predict" or arguments.command == "train" or arguments.command == "validate":
        run_prediction()
    elif arguments.command == "settings":
        run_settings()


if __name__ == "__main__":
    settings = Settings()
    models_info = ModelsInfo()
    main()
