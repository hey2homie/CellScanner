import os
import yaml
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def set_tf_hardware(hardware: str) -> None:
    """
    Set the hardware to use for the tensorflow session, which can be either CPU or GPU.
    Args:
        hardware (str): The hardware to use for the tensorflow session, which can be either CPU or GPU.
    Returns:
        None.
    Raises:
        ValueError: If the hardware is not available.
        RuntimeError: If TensorFlow is already initialized.
    """
    try:
        tf.config.set_visible_devices([], hardware)
    except ValueError:
        raise Exception(hardware + "not available")
    except RuntimeError:
        raise Exception("TensorFlow is already running")


def get_available_models_fc(models: dict, fc: str) -> list:
    """
    For a given flow cytometer, return list of available models. Can be either list of autoencoders or classifiers.
    Args:
        models (dict): Dictionary of models.
        fc (str): Flow cytometer.
    Returns:
        available_models (list): List of available models.
    """
    available_models = []
    for key, value in models.items():
        if value[0]["fc_type"] == fc.split("_")[1].capitalize():
            available_models.append(key)
    return available_models


def get_available_cls(classifiers: dict, ae: str) -> list:
    """
    For a given autoencoder, return list of compatible classifiers.
    Args:
        classifiers (dict): Dictionary of classifiers.
        ae (str): Name of the autoencoder.
    Returns:
        available_models (list): List of available classifiers.
    """
    available_models = []
    for key, value in classifiers.items():
        if value[-1]["autoencoder"] == ae:
            available_models.append(key)
    return available_models


def create_output_dir(path: str) -> str:
    """
    Create a directory for the output, with today's date as folder name.
    Args:
        path (str): Path to the output directory.
    Returns:
        output_path (str): Path to the output directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    output_path = path + "/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "/"
    os.mkdir(output_path)
    return output_path


def save_cell_counts(path: str, inputs: dict, mse_threshold: float, prob_threshold: float) -> None:
    """
    Save the prediction results to the txt files.
    Args:
        path (str): Path to the output directory.
        inputs (dict): Dictionary of inputs.
        mse_threshold (float): Threshold for the MSE.
        prob_threshold (float): Threshold for the probability.
    Returns:
        None
    """
    with open(path + "cell_counts.txt", "w") as file:
        output = {}
        for key, value in inputs.items():
            output[key] = {}
            labels = inputs[key]["labels"]
            mse = inputs[key]["mse"]
            probs = inputs[key]["probability_pred"]
            labels_count = np.unique(labels, return_counts=True)
            labels_count = dict(zip(labels_count[0], labels_count[1]))
            all_cells = sum(labels_count.values())
            for label in labels_count.keys():
                indices = np.where(labels == label)[0]
                mse_label = mse[indices]
                blanks_label = len(np.where(mse_label > mse_threshold)[0])
                probs_label = probs[indices]
                probs_label = len(np.where(probs_label > prob_threshold)[0])
                percentage_cells = np.round(labels_count[label] / all_cells * 100, 2)
                percentage_blanks = np.round(blanks_label / len(indices) * 100, 2)
                percentage_probs = np.round(probs_label / len(indices) * 100, 2)
                output[key][str(label)] = {"Number of cells": int(labels_count[label]),
                                           "Percentage": float(percentage_cells),
                                           "Number of blanks": int(blanks_label),
                                           "Percentage of blanks": float(percentage_blanks),
                                           "Number of high probability": int(probs_label),
                                           "Percentage of high probability": float(percentage_probs),
                                           "Results after gating": int(labels_count[label]) - int(blanks_label)}
        yaml.dump(output, file, default_flow_style=False, sort_keys=False)


def get_plotting_info(settings, data: np.ndarray) -> Tuple[list, list]:
    """
    Get the plotting information (axis content, axis names) for the scatter plots.
    Args:
        settings (Settings): Settings object.
        data (np.ndarray): Data used for plotting.
    Returns:
        dataframe (list): List containing lists representing the data for each axis.
        names (list): List containing the names of the axes.
    """
    from .settings import SettingsOptions
    if settings.vis_type == "UMAP":
        names = ["X", "Y", "Z"]
        dataframe = data["embeddings"]
        dataframe = [dataframe[:, index] for index in range(dataframe.shape[1])]
    else:
        if settings.fc_type == "Accuri":
            available_channels = SettingsOptions.vis_channels_accuri.value
            channels_to_use = settings.vis_channels_accuri
        else:
            available_channels = SettingsOptions.vis_channels_cytoflex.value
            channels_to_use = settings.vis_channels_cytoflex
        indexes = [available_channels.index(channel) for channel in channels_to_use]
        names = [available_channels[index] for index in indexes]
        dataframe = [data["data"][:, index] for index in indexes]
    return dataframe, names


def create_dataframe_vis(settings, data: dict, data_vis: list, names: list) -> Tuple[pd.DataFrame, str]:
    """
    Create a dataframe for the scatter plots used for the visualisations.
    Args:
        settings (Settings): Settings object.
        data (dict): Dictionary containing all additional data apart from main dataframe with channels content.
        data_vis (list): List containing lists representing the data for each axis.
        names (list): List containing the names of the axes.
    Returns:
        dataframe (pd.DataFrame): Dataframe containing the data for the scatter plots.
        color (str): Name of the column that will be used for color differentiation: either predicted labels or
        correct/incorrect label.
    """
    if settings.vis_dims == 2:
        dataframe = pd.DataFrame({names[0]: data_vis[0].astype(np.float32),
                                  names[1]: data_vis[1].astype(np.float32)})
    else:
        dataframe = pd.DataFrame({names[0]: data_vis[0].astype(np.float32),
                                  names[1]: data_vis[1].astype(np.float32),
                                  names[2]: data_vis[2].astype(np.float32)})
    dataframe["Species"] = data["labels"].astype(str)
    dataframe["Probability"] = data["probability_best"].astype(np.float32)
    dataframe["MSE"] = data["mse"].astype(np.float32)
    try:
        dataframe["Labels"] = data["true_labels"].astype(str)
        dataframe["Correctness"] = data["labels_compared"].astype(str)
        color = "Correctness"
    except KeyError:
        color = "Species"
    if len(dataframe) > 20000:
        dataframe = dataframe.sample(n=20000, random_state=42)
    return dataframe, color


def match_blanks_to_mse(predicted_labels: np.ndarray, mse: np.ndarray, threshold: float) -> np.ndarray:
    """
    Labels events as blanks if the corresponding MSE is above the threshold.
    Args:
        predicted_labels (np.ndarray): Predicted labels.
        mse (np.ndarray): Mean squared error for each event.
        threshold (float): Threshold for the MSE.
    Returns:
        predicted_labels (np.ndarray): Predicted labels with outliers labelled as blanks.
    """
    np.place(predicted_labels, mse > threshold, "Blank")
    return predicted_labels


def drop_blanks(true_labels: np.ndarray, predicted_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop blanks from the true labels and predicted probabilities.
    Args:
        true_labels (np.ndarray): True labels.
        predicted_probs (np.ndarray): Predicted probabilities.
    Returns:
        true_labels (np.ndarray): True labels without blanks.
        predicted_probs (np.ndarray): Predicted probabilities without blanks.
    """
    indices = np.where(true_labels != "Blank")[0]
    true_labels = true_labels[indices]
    predicted_labels = predicted_probs[indices]
    return true_labels, predicted_labels
