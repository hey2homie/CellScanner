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

    def __make_plots(self, cols: list) -> None:
        if self.settings.vis_dims == "3":
            try:
                self.graph_outputs = scatter_3d(self.data, x=cols[0], y=cols[1], z=cols[2], color="Species")
            except ValueError:
                self.graph_outputs = scatter_3d(self.data, x=cols[0], y=cols[1], z=cols[2], color="Correctness")
        else:
            try:
                self.graph_outputs = scatter(self.data, x=cols[0], y=cols[1], color="Species")
            except ValueError:
                self.graph_outputs = scatter(self.data, x=cols[0], y=cols[1], color="Correctness")
        self.graph_outputs.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                                     font=dict(family="Avenir", size=8, color="black")))
        try:
            self.graph_mse_err = scatter(self.data, x=self.data.index, y="MSE", color="Species")
        except ValueError:
            self.graph_mse_err = scatter(self.data, x=self.data.index, y="MSE", color="Correctness")
        self.graph_mse_err.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                                     font=dict(family="Avenir", size=8, color="black")))
        self.graph_mse_err.add_hline(y=self.settings.mse_threshold, line_color="red")

    def __configurate_widgets(self) -> None:
        self.layout_graph.setContentsMargins(0, 0, 0, 0)
        self.file_box.currentItemChanged.connect(lambda: self.set_inputs())

    def set_items(self, items: list) -> None:
        self.file_box.addItems(items)
        self.file_box.setCurrentItem(self.file_box.item(0))

    def set_inputs(self, inputs: dict = None) -> None:
        if self.file_box.count() == 0:
            self.file_box.hide()
            self.children()[3].hide()
            self.children()[4].hide()
            self.widget_graph.setGeometry(*[46, 43, 808, 450])
            pass
        if inputs:
            self.inputs = inputs
            value = list(self.inputs.keys())[0]
        else:
            try:
                value = self.file_box.currentItem().text()
            except AttributeError:
                return None
        if self.inputs:
            self.data, mse = self.inputs[value][0], self.inputs[value][1]
            if self.settings.vis_type == "UMAP":
                if self.settings.vis_dims == "3":
                    self.data = pd.DataFrame({"X": self.data[:, -3].astype(np.float32),
                                              "Y": self.data[:, -2].astype(np.float32),
                                              "Z": self.data[:, -1].astype(np.float32),
                                              "Species": self.data[:, -5].astype(np.str),
                                              "Probability": self.data[:, -4].astype(np.float32)})
                    names = ["X", "Y", "Z"]
                else:
                    self.data = pd.DataFrame({"X": self.data[:, -1].astype(np.float32),
                                              "Y": self.data[:, -2].astype(np.float32),
                                              "Species": self.data[:, -4].astype(np.str),
                                              "Probability": self.data[:, -3].astype(np.float32)})
                    names = ["X", "Y"]
            else:
                if self.settings.fc_type == "Accuri":
                    available_channels = SettingsOptions.vis_channels_accuri.value
                    channels_to_use = self.settings.vis_channels_accuri
                else:
                    available_channels = SettingsOptions.vis_channels_cytoflex.value
                    channels_to_use = self.settings.vis_channels_cytoflex
                indexes = [available_channels.index(channel) for channel in channels_to_use]
                names = [available_channels[index] for index in indexes]
                if self.settings.vis_dims == "3":
                    self.data = pd.DataFrame({names[0]: self.data[:, indexes[0]].astype(np.float32),
                                              names[1]: self.data[:, indexes[1]].astype(np.float32),
                                              names[2]: self.data[:, indexes[2]].astype(np.float32),
                                              "Species": self.data[:, -2].astype(np.str),
                                              "Probability": self.data[:, -1].astype(np.float32)})
                else:
                    self.data = pd.DataFrame({names[0]: self.data[:, indexes[0]].astype(np.float32),
                                              names[1]: self.data[:, indexes[1]].astype(np.float32),
                                              "Species": self.data[:, -2].astype(np.str),
                                              "Probability": self.data[:, -1].astype(np.float32)})
            # TODO: The whole different type/dims is super ugly. Refactor this.
            self.data["MSE"] = mse.astype(np.float32)
            if any(species in self.data["Species"].unique() for species in ["Correct", "Incorrect"]):
                self.data.rename(columns={"Species": "Correctness"}, inplace=True)
            self.__make_plots(cols=names)
            self.children()[3].setText("MSE")
            self.browser.setHtml(self.graph_outputs.to_html(include_plotlyjs="cdn"))

    def change_plot(self, plot_type: str) -> None:
        if plot_type == "MSE":
            self.browser.setHtml(self.graph_mse_err.to_html(include_plotlyjs="cdn"))
            self.children()[3].setText("Predictions")
        elif plot_type == "Predictions":
            self.children()[3].setText("MSE")
            self.browser.setHtml(self.graph_outputs.to_html(include_plotlyjs="cdn"))

    def adjust_mse(self) -> None:
        mse, entered = InputDialog.getText(self, "", "Enter new MSE threshold")
        if entered:
            try:
                self.settings.mse_threshold = float(mse)
                self.graph_mse_err.update_shapes(dict(y0=self.settings.mse_threshold, y1=self.settings.mse_threshold))
                self.browser.setHtml(self.graph_mse_err.to_html(include_plotlyjs="cdn"))
            except ValueError:
                print("Invalid MSE threshold")

    def save_results(self) -> None:
        visualizations = MplVisualization(output_path=self.settings.results)
        visualizations.save_predictions_visualizations(inputs=self.inputs, settings=self.settings)
        # TODO: Save data here

    def clear(self) -> None:
        self.inputs = None
        self.data = None
        self.graph_outputs = None
        self.graph_mse_err = None
        self.file_box.clear()
        self.browser.setHtml("")


class ResultsTraining(Widget):

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
        self.tf_board = subprocess.Popen(["tensorboard", "--logdir=.tflogs/" + name, "--port=6006"])
        self.browser.load(QUrl("http://localhost:6006/#scalars"))
        self.browser.reload()
        self.widget_browser.layout().addWidget(self.browser)
