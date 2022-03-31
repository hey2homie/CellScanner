from PyQt6.QtWidgets import QWidget
import pyqtgraph.opengl as gl
import numpy as np
import pandas as pd


class Scatter3D(QWidget):

    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.widget = gl.GLViewWidget()
        self.grid = gl.GLGridItem()
        self.data = data.iloc[:, :3]
        self.labels = data.iloc[:, 4]
        self.__unit_ui()

    def __unit_ui(self):
        self.widget.addItem(self.grid)
        self.__add_input()
        self.widget.show()

    def __add_input(self):
        data = self.data.to_numpy()
        self.widget.addItem(gl.GLScatterPlotItem(pos=data))
