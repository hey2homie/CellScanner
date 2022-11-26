import subprocess

from PyQt6 import QtWebEngineWidgets
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QGridLayout

from plotly.express import scatter_3d, scatter
import numpy as np
import pandas as pd

from utilities.visualizations import MplVisualization
from utilities.settings import SettingsOptions

from gui.widgets import Widget, Button, FileBox, InputDialog


class ResultsClassification(Widget):
    """
    Attributes:
    ----------
    inputs: dict
        Dictionary containing the pairs file : results of predictions.
    data: pd.DataFrame
        Dataframe containing the results of the predictions for the current file
    widget_graph: Widget
        Widgets containing web browser to display the plots.
    graph_outputs: plotly.graph_objects.Figure
        Plotly figure containing the predictions for the current file.
    graph_mse_err: plotly.graph_objects.Figure
        Plotly figure containing the reconstruction error for the current file.
    layout_graph: QGridLayout
        Layout of the widget_graph.
    browser: QtWebEngineWidgets.QWebEngineView
        Web browser to display the plots.
    file_box: FileBox
        File box to select the file to display.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.settings = self.stack.widget(0).settings
        self.inputs = None
        self.data = None
        self.widget_graph = None
        self.graph_outputs = None
        self.graph_mse_err = None
        self.layout_graph = None
        self.browser = QtWebEngineWidgets.QWebEngineView(parent=None)
        self.file_box = None
        self.__init_ui()

    def __init_ui(self) -> None:
        self.setWindowTitle("Results")
        self.__init_widgets()
        self.__configurate_widgets()

    def __init_widgets(self) -> None:
        self.widget_graph = Widget(obj_name="", geometry=[254, 43, 595, 450], parent=self)
        self.layout_graph = QGridLayout(parent=self.widget_graph)
        self.layout_graph.addWidget(self.browser)
        self.file_box = FileBox(obj_name="select", geometry=[46, 43, 180, 450], parent=self)
        Button(text="Menu", obj_name="standard", geometry=[46, 518, 180, 60], parent=self)
        Button(text="MSE", obj_name="standard", geometry=[254, 518, 180, 60], parent=self)
        Button(text="Adjust MSE", obj_name="standard", geometry=[462, 518, 180, 60], parent=self)
        Button(text="Save Results", obj_name="standard", geometry=[669, 518, 180, 60], parent=self)

    def __configurate_widgets(self) -> None:
        self.layout_graph.setContentsMargins(0, 0, 0, 0)
        self.file_box.currentItemChanged.connect(lambda: self.set_inputs())

    def set_items(self, items: list) -> None:
        """
        Adds items in the file box.
        Args:
            items (list): List of items to add.
        Returns:
            None.
        """
        self.file_box.addItems(items)
        self.file_box.setCurrentItem(self.file_box.item(0))

    def set_inputs(self, diagnostics: bool = False) -> None:
        """
        Sets the inputs used to display plots and save the information. After running predictions, the default plot to
        display will be the one at index 0 in the file box. After setting the inputs, for default constructs the Pandas
        dataframe which is used as the source for the plots. Changing files in the file box updates this dataframe.
        Args:
            diagnostics (bool, optional): If True, the color scheme displays correct/incorrect predictions.
        Returns:
            None.
        """
        if not self.inputs:
            return None
        if self.file_box.count() == 0:
            self.file_box.hide()
            self.children()[5].hide()
            self.widget_graph.setGeometry(46, 43, 808, 450)
            current_item = self.inputs
        else:
            value = self.file_box.currentItem().text()
            current_item = self.inputs[value]
        color = "Species"
        if self.settings.vis_type == "UMAP":
            names = ["X", "Y", "Z"]
            dataframe = current_item["embeddings"]
            dataframe = [dataframe[:, index] for index in range(dataframe.shape[1])]
        else:
            if self.settings.fc_type == "Accuri":
                available_channels = SettingsOptions.vis_channels_accuri.value
                channels_to_use = self.settings.vis_channels_accuri
            else:
                available_channels = SettingsOptions.vis_channels_cytoflex.value
                channels_to_use = self.settings.vis_channels_cytoflex
            indexes = [available_channels.index(channel) for channel in channels_to_use]
            names = [available_channels[index] for index in indexes]
            dataframe = [current_item["data"][:, index] for index in indexes]
        self.data = pd.DataFrame({names[0]: dataframe[0].astype(np.float32), names[1]: dataframe[1]})
        self.data["Species"] = current_item["labels"].astype(str)
        if diagnostics:
            self.data["Correctness"] = current_item["labels_compared"].astype(str)
            color = "Correctness"
        self.graph_outputs = scatter(self.data, x=names[0], y=names[1], color=color)
        if self.settings.vis_dims == 3:
            self.data[names[2]] = dataframe[2].astype(np.float32)
            self.graph_outputs = scatter_3d(self.data, x=names[0], y=names[1], z=names[2], color=color)
        self.data["Probability"] = current_item["probability_best"].astype(np.float32)
        self.data["MSE"] = current_item["mse"].astype(np.float32)
        layout_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                             font=dict(family="Avenir", size=8, color="black"))
        self.graph_outputs.update_layout(legend=layout_legend)
        self.graph_mse_err = scatter(self.data, x=self.data.index, y="MSE", color="Species")
        self.graph_mse_err.update_layout(legend=layout_legend)
        self.graph_mse_err.add_hline(y=self.settings.mse_threshold, line_color="red")
        self.children()[3].setText("MSE")
        self.browser.setHtml(self.graph_outputs.to_html(include_plotlyjs="cdn"))

    def change_plot(self, plot_type: str) -> None:
        """
        Displays the selected plot type in the browser.
        Args:
            plot_type (str): The plot type to display.
        Returns:
            None.
        """
        if plot_type == "MSE":
            self.browser.setHtml(self.graph_mse_err.to_html(include_plotlyjs="cdn"))
            self.children()[3].setText("Predictions")
        elif plot_type == "Predictions":
            self.children()[3].setText("MSE")
            self.browser.setHtml(self.graph_outputs.to_html(include_plotlyjs="cdn"))

    def adjust_mse(self) -> None:
        """
        Adjusts the MSE threshold and saves the new value to the settings class.
        Returns:
            None.
        """
        mse, entered = InputDialog.getText(self, "", "Enter new MSE threshold")
        if entered:
            try:
                self.settings.mse_threshold = float(mse)
                self.graph_mse_err.update_shapes(dict(y0=self.settings.mse_threshold, y1=self.settings.mse_threshold))
                self.browser.setHtml(self.graph_mse_err.to_html(include_plotlyjs="cdn"))
            except ValueError:
                print("Invalid MSE threshold")

    def save_results(self) -> None:
        """
        Saves the results of the predictions for each file.
        Returns:
            None.
        """
        visualizations = MplVisualization(self.settings.results)
        visualizations.save_predictions_visualizations(self.inputs, self.settings)

    def clear(self) -> None:
        """
        Clears the content of class.
        Returns:
            None.
        """
        self.inputs = None
        self.data = None
        self.graph_outputs = None
        self.graph_mse_err = None
        self.file_box.clear()
        self.browser.setHtml("")
        self.file_box.show()
        self.children()[5].show()
        self.widget_graph.setGeometry(254, 43, 595, 450)


class ResultsTraining(Widget):
    """
    Attributes:
    ----------
    tf_board: subprocess.Popen
        Tensorboard process running separately from the application.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.browser = QtWebEngineWidgets.QWebEngineView(parent=None)
        self.tf_board = None
        self.__init_ui()

    def __init_ui(self) -> None:
        self.setWindowTitle("Results")
        self.__init_widgets()

    def __init_widgets(self) -> None:
        self.widget_browser = Widget(obj_name="", geometry=[46, 43, 808, 450], parent=self)
        self.layout_graph = QGridLayout(parent=self.widget_browser)
        self.layout_graph.addWidget(self.browser)
        Button(text="Menu", obj_name="standard", geometry=[46, 518, 200, 60], parent=self)

    def run_tf_board(self, name: str) -> None:
        """
        Runs TensorBoard as a separate process.
        Args:
            name: Name of the TensorBoard log directory.
        Returns:
            None.
        """
        self.tf_board = subprocess.Popen(["tensorboard", "--logdir=training_logs/" + name, "--port=6006"])
        self.browser.load(QUrl("http://localhost:6006/#scalars"))
        self.browser.reload()
        self.widget_browser.layout().addWidget(self.browser)
        self.browser.reload()
        self.browser.reload()
