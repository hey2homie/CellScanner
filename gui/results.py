import subprocess

from PyQt6 import QtWebEngineWidgets
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QGridLayout

from plotly.express import scatter_3d, scatter

from utilities.visualizations import MplVisualization
from utilities.helpers import create_output_dir, save_cell_counts, get_plotting_info, create_dataframe_vis

from .widgets import Widget, Button, FileBox, InputDialog, MessageBox


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
        Button(text="Save Results", obj_name="standard", geometry=[669, 518, 180, 60], parent=self)
        Button(text="MSE", obj_name="standard", geometry=[254, 518, 180, 60], parent=self)
        Button(text="Adjust MSE", obj_name="standard", geometry=[462, 518, 180, 60], parent=self)

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
        if self.settings.gating_type == "Autoencoder":
            self.__show_widgets([4, 5])
        else:
            self.__hide_widgets([4, 5])

    def set_inputs(self) -> None:
        """
        Sets the inputs used to display plots and save the information. After running predictions, the default plot to
        display will be the one at index 0 in the file box. After setting the inputs, for default constructs the Pandas
        dataframe which is used as the source for the plots. Changing files in the file box updates this dataframe.
        Returns:
            None.
        """
        if not self.inputs:
            return None
        if self.file_box.count() == 0:
            self.file_box.hide()
            self.children()[3].hide()
            self.widget_graph.setGeometry(46, 43, 808, 450)
            current_item = self.inputs
        else:
            value = self.file_box.currentItem().text()
            current_item = self.inputs[value]
        dataframe, names = get_plotting_info(self.settings, current_item)
        self.data, color = create_dataframe_vis(self.settings, current_item, dataframe, names)
        self.__make_plot(names, color)
        if self.settings.gating_type == "Autoencoder":
            self.children()[4].setText("MSE")
        self.browser.setHtml(self.graph_outputs.to_html(include_plotlyjs="cdn"))

    def __make_plot(self, names: list, color: str) -> None:
        """
        Creates the plots for the current file. Can be either a 3D scatter plot or a 2D scatter plot. Also, creates plot
        for the MSE reconstruction error.
        Args:
            names (list): List of names of the columns in the dataframe.
            color (str): Name of the column to use as the color for the plots.
        Returns:
            None.
        """
        if self.settings.vis_dims == 2:
            self.graph_outputs = scatter(self.data, x=names[0], y=names[1], color=color)
        else:
            self.graph_outputs = scatter_3d(self.data, x=names[0], y=names[1], z=names[2], color=color)
        layout_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                             font=dict(family="Avenir", size=8, color="black"))
        self.graph_outputs.update_layout(legend=layout_legend)
        if self.settings.gating_type == "Autoencoder":
            color_mse = "Labels" if color == "Correctness" else "Species"
            self.graph_mse_err = scatter(self.data, x=self.data.index, y="Gating_results", color=color_mse)
            self.graph_mse_err.update_layout(legend=layout_legend)
            self.graph_mse_err.add_hline(y=self.settings.mse_threshold, line_color="red")

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
            self.children()[4].setText("Predictions")
        elif plot_type == "Predictions":
            self.children()[4].setText("MSE")
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
                self.stack.widget(4).set_values_from_config()
                self.stack.widget(4).update_config()
            except ValueError:
                MessageBox.about(self, "Warning", "Invalid MSE threshold")

    def save_results(self) -> None:
        """
        Saves the results of the predictions for each file.
        Returns:
            None.
        """
        output_dir = create_output_dir(path=self.settings.results)
        visualizations = MplVisualization(output_path=output_dir)
        visualizations.save_predictions_visualizations(inputs=self.inputs, settings=self.settings)
        save_cell_counts(path=output_dir, inputs=self.inputs, gating_type=self.settings.gating_type,
                         mse_threshold=self.settings.mse_threshold, prob_threshold=self.settings.softmax_prob_threshold)

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
        self.children()[3].show()
        self.widget_graph.setGeometry(254, 43, 595, 450)

    def __hide_widgets(self, widgets: list) -> None:
        """
        Hides the widgets at the given indices.
        Args:
            widgets (list): List of indices of widgets to hide.
        Returns:
            None.
        """
        for child in widgets:
            self.children()[child].hide()

    def __show_widgets(self, widgets: list) -> None:
        """
        Shows the widgets at the given indices.
        Args:
            widgets (list): List of indices of widgets to show.
        Returns:
            None.
        """
        for child in widgets:
            self.children()[child].show()


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
        self.widget_browser = Widget(obj_name="", geometry=[46, 43, 993, 550], parent=self)
        self.layout_graph = QGridLayout(parent=self.widget_browser)
        self.layout_graph.addWidget(self.browser)
        Button(text="Menu", obj_name="standard", geometry=[46, 618, 200, 60], parent=self)

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
        self.widget_browser.layout().addWidget(self.browser)
        self.browser.reload()
