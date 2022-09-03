from PyQt6 import QtWebEngineWidgets
from PyQt6.QtWidgets import QWidget, QStackedWidget, QGridLayout
import plotly.express as px
import numpy as np
import pandas as pd

from utilities.settings import Settings
from utilities.visualizations import MplVisualization


class ResultsWindow(QWidget):

    def __init__(self, stack: QStackedWidget, settings: Settings, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.settings = settings
        self.inputs = None
        self.data = None
        self.widget_graph = None
        self.layout_graph = None
        self.browser = QtWebEngineWidgets.QWebEngineView(parent=None)
        self.file_box = None
        self.__init_ui()

    def __init_ui(self) -> None:
        self.setWindowTitle("Results")
        self.setGeometry(self.stack.currentWidget().geometry())
        self.__init_widgets()
        self.__configurate_widgets()
        self.stack.addWidget(self)

    def __init_widgets(self) -> None:
        from widgets import Widget, Button, Styles, FileBox
        self.widget_graph = Widget(widget=Styles.Widget, obj_name="", geometry=[273, 43, 581, 450], parent=self)
        self.layout_graph = QGridLayout(parent=self.widget_graph)
        self.layout_graph.addWidget(self.browser)
        self.file_box = FileBox(obj_name="select", geometry=[47, 43, 200, 450], parent=self)
        Button(text="Menu", obj_name="standard", geometry=[46, 518, 200, 60], parent=self)
        Button(text="Save Data", obj_name="standard", geometry=[349, 518, 200, 60], parent=self)
        Button(text="Save Visuals", obj_name="standard", geometry=[652, 518, 200, 60], parent=self)

    def __init_graph(self) -> None:
        if self.settings.vis_dims == "3D":
            try:
                fig = px.scatter_3d(self.data, x="X", y="Y", z="Z", color="Species")
            except ValueError:
                fig = px.scatter_3d(self.data, x="X", y="Y", z="Z", color="Correctness")
        else:
            try:
                fig = px.scatter_3d(self.data, x="X", y="Y", color="Species")   # TODO: Change to 2D
            except ValueError:
                fig = px.scatter_3d(self.data, x="X", y="Y", color="Correctness")
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                      font=dict(family="Avenir", size=8, color="black")))
        # TODO: Add plot's tittle
        self.browser.setHtml(fig.to_html(include_plotlyjs="cdn"))

    def __configurate_widgets(self) -> None:
        self.layout_graph.setContentsMargins(0, 0, 0, 0)
        self.file_box.currentItemChanged.connect(lambda: self.set_inputs())

    def set_items(self, items: list) -> None:
        # items = [item.split("/")[-1] for item in items]   # TODO: Remove the path to the files (if needed)
        self.file_box.addItems(items)
        self.file_box.setCurrentItem(self.file_box.item(0))

    def set_inputs(self, inputs: dict = None) -> None:
        if self.file_box.count() == 0:
            self.file_box.hide()
            self.children()[3].hide()
            self.children()[4].hide()
            self.widget_graph.set_geometry([46, 43, 808, 450])
            pass
        if inputs:
            self.inputs = inputs
            value = list(self.inputs.keys())[0]
        else:
            value = self.file_box.currentItem().text()
        if self.inputs:
            self.data = self.inputs[value]
            self.data = pd.DataFrame({"X": self.data[:, -4].astype(np.float32),
                                      "Y": self.data[:, -3].astype(np.float32),
                                      "Z": self.data[:, -2].astype(np.float32),
                                      "Species": self.data[:, -1]})
            if any(species in self.data["Species"].unique() for species in ["Correct", "Incorrect"]):
                self.data.rename(columns={"Species": "Correctness"}, inplace=True)
            self.__init_graph()

    def save_visuals(self) -> None:
        visualizations = MplVisualization(output_path=self.settings.results)
        visualizations.save_predictions_visualizations(inputs=self.inputs, settings=self.settings)
