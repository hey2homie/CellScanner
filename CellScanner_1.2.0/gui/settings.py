import os
from enum import Enum

from PyQt6.QtWidgets import QWidget, QStackedWidget
from PyQt6.QtCore import Qt

from utilities.settings import Settings

from gui.widgets import Widget, Styles, Button, HLine, ComboBox, CheckableComboBox, EditLine, Label


class ComboBoxContent(Enum):
    """
    Enum containing options for the Combo Boxes used in SettingsWindow.
    """
    FC = iter(["Accuri", "Cytoflex"])
    Classifiers = iter(os.listdir("./classifiers/"))
    VisType = iter(["UMAP", "Channels"])
    Dims = iter(["2D", "3D"])
    Channels_Accuri = ["FL1-A", "FL2-A", "FL3-A", "FL4-A", "FSC-H", "SSC-H", "FL1-H", "FL2-H", "FL3-H", "FL4-H",
                       "Width", "Time"]
    Channels_CytoFlex = ["FSC-H", "FSC-A", "SSC-H", "SSC-A", "FL1-H", "FL1-A", "FL4-H", "FL4-A", "FL3-red-H",
                         "FL3-red-A", "APC-A750-H", "APC-A750-A", "VSSC-H", "VSSC-A", "KO525-H", "KO525-A",
                         "FL2-orange-H", "FL2-orange-A", "mCherry-H", "mCherry-A", "PI-H", "PI-A", "FSC-Width", "Time"]
    Hardware = ["GPU", "CPU"]
    Lr = ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"]
    Lr_reduced = ["1e-3", "1e-2"]
    LrScheduler = ["Constant", "Time Decay", "Step Decay", "Exponential Decay"]
    Gating = ["Line", "Autoencoder", "Machine"]

    @property
    def get_value(self) -> list:
        return iter(self.value)


class SettingsWindow(QWidget):
    fc_type = None
    model = None
    results = None
    vis_type = None
    vis_channels_accuri = None
    vis_channels_cytoflex = None
    vis_dims = None
    num_umap_cores = None
    hardware = None
    num_batches = None
    num_epochs = None
    lr = None
    lr_scheduler = None
    lr_reduced = None
    cols_to_drop_accuri = None
    cols_to_drop_cytoflex = None
    gating_type = None
    autoencoder = None
    mse_threshold = None  # TODO: Add adjustment to the widget
    # MSE is logarithmic value !!!

    def __init__(self, stack: QStackedWidget, settings: Settings, *args, **kwargs) -> None:
        """
        Args:
            stack (QStackedWidget): The stacked widget containing the all application's windows.
            settings (Settings): The settings class.
            *args: Optional arguments of the QWidget class.
            **kwargs: Optional arguments of the QWidget class.
        """
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.widget_settings = None
        self.widget_general = None
        self.widget_vis = None
        self.widget_nn = None
        self.widget_data = None
        self.settings = settings
        self.attributes = {}
        self.__set_attributes()
        self.__init_ui()

    def __set_attributes(self) -> None:
        """
        Copies class attributes from settings class
        Returns:
            None
        """
        # TODO: Try a way not to repeat code from settings
        # TODO: Preferably remove all class attributes and operate only with settings attributes.
        for key, value in self.settings.get_attributes().items():
            setattr(self, key, value)
            self.attributes[key] = value

    def __init_ui(self) -> None:
        self.setWindowTitle("Settings")
        self.setGeometry(self.stack.currentWidget().geometry())
        self.__init_widgets()
        self.__init_elements()
        self.__set_values_from_config()
        self.stack.addWidget(self)

    def __init_widgets(self) -> None:
        # TODO: See below.
        # So what needs to be done in here. All the settings should be filled in from the config. I need to allow for
        # multiple choices in some combo-boxes (namely in channels for visualisations). Definitely need to come up with
        # more clever way to organise all the labels, box-combo and input fields, maybe with the grids with custom
        # geometry of cells. Other interface related issues, but that's for later.
        # TODO: Set objects names according to style sheet
        self.widget_settings = Widget(widget=Styles.Widget, obj_name="settings", geometry=[46, 90, 808, 400],
                                      parent=self)
        self.widget_general = Widget(widget=Styles.Widget, obj_name="settings", geometry=[0, 0, 808, 100],
                                     parent=self.widget_settings)
        self.widget_vis = Widget(widget=Styles.Widget, obj_name="settings", geometry=[0, 100, 808, 100],
                                 parent=self.widget_settings)
        self.widget_nn = Widget(widget=Styles.Widget, obj_name="settings", geometry=[0, 200, 808, 100],
                                parent=self.widget_settings)
        self.widget_data = Widget(widget=Styles.Widget, obj_name="settings", geometry=[0, 300, 808, 100],
                                  parent=self.widget_settings)
        Button(text="Menu", obj_name="standard", geometry=[46, 518, 200, 60], parent=self)
        Button(text="Apply", obj_name="standard", geometry=[652, 518, 200, 60], parent=self)

    def __init_elements(self) -> None:

        def scheduler_options_show(widget: ComboBox) -> None:
            if widget.currentIndex() != 0:
                self.widget_nn.children()[11].show()
                self.widget_nn.children()[12].show()
            else:
                self.widget_nn.children()[11].hide()
                self.widget_nn.children()[12].hide()

        def channels_show(widget: ComboBox) -> None:
            if widget.currentIndex() == 1:
                self.widget_vis.children()[7].show()
                self.widget_vis.children()[8].show()
            else:
                self.widget_vis.children()[7].hide()
                self.widget_vis.children()[8].hide()

        def fc_changed(fc_widget: ComboBox) -> None:
            self.update_config()
            self.widget_vis.children()[8].clear()
            self.widget_data.children()[4].clear()
            if fc_widget.currentIndex() == 0:
                self.widget_vis.children()[8].set_name("vis_channels_accuri")
                self.widget_vis.children()[8].addItems(ComboBoxContent.Channels_Accuri.get_value)
                self.widget_vis.children()[8].set_checked_items(self.vis_channels_accuri)
                self.widget_data.children()[4].set_name("cols_to_drop_accuri")
                self.widget_data.children()[4].addItems(ComboBoxContent.Channels_Accuri.get_value)
                self.widget_data.children()[4].set_checked_items(self.cols_to_drop_accuri)
            else:
                self.widget_vis.children()[8].set_name("vis_channels_cytoflex")
                self.widget_vis.children()[8].addItems(ComboBoxContent.Channels_CytoFlex.get_value)
                self.widget_vis.children()[8].set_checked_items(self.vis_channels_cytoflex)
                self.widget_data.children()[4].set_name("cols_to_drop_cytoflex")
                self.widget_data.children()[4].addItems(ComboBoxContent.Channels_CytoFlex.get_value)
                self.widget_data.children()[4].set_checked_items(self.cols_to_drop_cytoflex)

        tittle_label = Label(text="Settings", obj_name="tittle", geometry=[0, 9, 895, 69], parent=self)
        general_label = Label(text="General", obj_name="settings", geometry=[0, 0, 98, 41],
                              parent=self.widget_general)
        fc_label = Label(text="FC type:", obj_name="small", geometry=[28, 37, 70, 26], parent=self.widget_general)
        fc_choose = ComboBox(obj_name="combobox", geometry=[100, 37, 150, 26], name="fc_type",
                             parent=self.widget_general)
        fc_choose.addItems(ComboBoxContent.FC.get_value)
        fc_update_button = Button(text="Update FC", obj_name="settings", geometry=[260, 37, 100, 26],
                                  parent=self.widget_general)
        classifier_label = Label(text="Classifier:", obj_name="small", geometry=[370, 37, 79, 26],
                                 parent=self.widget_general)
        classifier_choose = ComboBox(obj_name="combobox", geometry=[440, 37, 270, 26], name="model",
                                     parent=self.widget_general)
        classifier_info = Button(text="?", obj_name="small", geometry=[720, 37, 26, 26], parent=self.widget_general)
        classifier_choose.addItems(ComboBoxContent.Classifiers.get_value)
        # TODO: Add action to update FC and add pop-up with info about classifier
        output_label = Label(text="Output location:", obj_name="small", geometry=[28, 70, 124, 26],
                             parent=self.widget_general)
        output_location_label = Label(text=self.results, obj_name="small", geometry=[160, 70, 604, 26], name="results",
                                      parent=self.widget_general)
        output_location_button = Button(text="...", obj_name="settings", geometry=[730, 70, 50, 26],
                                        parent=self.widget_general)
        HLine(obj_name="line", geometry=[100, 16, 705, 1], parent=self.widget_general)
        vis_label = Label(text="Visualizations", obj_name="settings", geometry=[0, 0, 153, 41],
                          parent=self.widget_vis)
        vis_type_label = Label(text="Visualisation type:", obj_name="small", geometry=[28, 37, 138, 26],
                               parent=self.widget_vis)
        vis_type_choose = ComboBox(obj_name="combobox", geometry=[141, 37, 110, 26], name="vis_type",
                                   parent=self.widget_vis)
        vis_type_choose.addItems(ComboBoxContent.VisType.get_value)
        # TODO: Set minimal items of 2 and maximum of 3 depending on the vis dims
        dim_label = Label(text="Dimension:", obj_name="small", geometry=[280, 37, 97, 26], parent=self.widget_vis)
        dim_choose = ComboBox(obj_name="combobox", geometry=[358, 37, 65, 26], name="vis_dims", parent=self.widget_vis)
        dim_choose.addItems(ComboBoxContent.Dims.get_value)
        umap_cores_label = Label(text="Number of cores to compute UMAP:", obj_name="small", geometry=[440, 37, 270, 26],
                                 parent=self.widget_vis)
        umap_cores_input = EditLine(obj_name="input", geometry=[670, 37, 30, 26], name="num_umap_cores",
                                    parent=self.widget_vis)
        channels_label = Label(text="Used channels:", obj_name="small", geometry=[28, 70, 123, 26],
                               parent=self.widget_vis)
        channels_choose = CheckableComboBox(obj_name="combobox", geometry=[130, 70, 90, 26], parent=self.widget_vis)
        channels_choose.addItems(ComboBoxContent.Channels_Accuri.get_value if self.fc_type == "Accuri" else
                                 ComboBoxContent.Channels_CytoFlex.get_value)
        HLine(obj_name="line", geometry=[155, 16, 705, 1], parent=self.widget_vis)
        nn_label = Label(text="Neural Network", obj_name="settings", geometry=[0, 0, 178, 41],
                         parent=self.widget_nn)
        hardware_label = Label(text="Hardware used for training:", obj_name="small", geometry=[28, 37, 203, 26],
                               parent=self.widget_nn)
        hardware_choose = ComboBox(obj_name="combobox", geometry=[199, 37, 75, 26], name="hardware",
                                   parent=self.widget_nn)
        hardware_choose.addItems(ComboBoxContent.Hardware.get_value)
        lr_label = Label(text="Learning rate:", obj_name="small", geometry=[291, 37, 108, 26], parent=self.widget_nn)
        lr_choose = ComboBox(obj_name="combobox", geometry=[384, 37, 85, 26], name="lr", parent=self.widget_nn)
        lr_choose.addItems(ComboBoxContent.Lr.get_value)
        lr_scheduler_label = Label(text="Learning rate scheduler:", obj_name="small", geometry=[494, 37, 182, 26],
                                   parent=self.widget_nn)
        lr_scheduler_choose = ComboBox(obj_name="combobox", geometry=[646, 37, 135, 26], name="lr_scheduler",
                                       parent=self.widget_nn)
        lr_scheduler_choose.addItems(ComboBoxContent.LrScheduler.get_value)
        batches_label = Label(text="Number of batches:", obj_name="small", geometry=[28, 70, 151, 26],
                              parent=self.widget_nn)
        batches_input = EditLine(obj_name="input", geometry=[150, 70, 50, 26], name="num_batches",
                                 parent=self.widget_nn)
        epochs_label = Label(text="Number of epochs:", obj_name="small", geometry=[238, 70, 147, 26],
                             parent=self.widget_nn)
        epochs_input = EditLine(obj_name="input", geometry=[360, 70, 50, 26], name="num_epochs", parent=self.widget_nn)
        lr_initial_label = Label(text="Initial learning rate:", obj_name="small", geometry=[444, 70, 128, 26],
                                 parent=self.widget_nn)
        lr_initial_choose = ComboBox(obj_name="combobox", geometry=[566, 70, 85, 26], name="lr_reduced",
                                     parent=self.widget_nn)
        lr_initial_choose.addItems(ComboBoxContent.Lr_reduced.get_value)
        HLine(obj_name="line", geometry=[180, 16, 705, 1], parent=self.widget_nn)
        data_label = Label(text="Data Preparation", obj_name="settings", geometry=[0, 0, 192, 41],
                           parent=self.widget_data)
        gating_label = Label(text="Gating type:", obj_name="small", geometry=[28, 37, 96, 26], parent=self.widget_data)
        gating_choose = ComboBox(obj_name="combobox", geometry=[118, 37, 110, 26], name="gating_type",
                                 parent=self.widget_data)
        gating_choose.addItems(ComboBoxContent.Gating.get_value)
        col_label = Label(text="Columns to drop:", obj_name="small", geometry=[238, 37, 105, 26], parent=self.widget_data)
        columns_to_drop = CheckableComboBox(obj_name="combobox", geometry=[360, 37, 130, 26], parent=self.widget_data)
        columns_to_drop.addItems(ComboBoxContent.Channels_Accuri.get_value if self.fc_type == "Accuri" else
                                 ComboBoxContent.Channels_CytoFlex.get_value)
        HLine(obj_name="line", geometry=[194, 16, 705, 1], parent=self.widget_data)
        # TODO: Make global alignment through iteration over children at the class initiation
        if vis_type_choose.currentIndex() == 0:
            channels_label.hide()
            channels_choose.hide()
            channels_choose.set_name("vis_channels_accuri")
            columns_to_drop.set_name("cols_to_drop_accuri")
            columns_to_drop.set_checked_items(self.cols_to_drop_accuri)
        else:
            channels_choose.set_name("vis_channels_cytoflex")
            columns_to_drop.set_name("cols_to_drop_cytoflex")
            columns_to_drop.set_checked_items(self.cols_to_drop_cytoflex)
        if lr_scheduler_choose.currentIndex() == 0:
            lr_initial_label.hide()
            lr_initial_choose.hide()
        lr_scheduler_choose.currentIndexChanged.connect(lambda: scheduler_options_show(lr_scheduler_choose))
        vis_type_choose.currentIndexChanged.connect(lambda: channels_show(vis_type_choose))
        fc_choose.currentIndexChanged.connect(lambda: fc_changed(fc_choose))
        tittle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        [label.setAlignment(Qt.AlignmentFlag.AlignLeft) for label in [general_label, vis_label, nn_label, data_label]]

    def __set_values_from_config(self) -> None:
        """
        Sets indexes of the combo-boxes and the input fields from the configuration file upon creating instance of the
        class.
        Returns:
            None
        """
        for widget in [self.widget_general, self.widget_vis, self.widget_nn, self.widget_data]:
            for child in widget.children():
                if isinstance(child, ComboBox) and child.name != "lr_initial":
                    items = [child.itemText(i) for i in range(child.count())]
                    # TODO: Add method to widget itself because it's used in multiple places
                    child.setCurrentIndex(items.index(self.attributes[child.name]))
                elif isinstance(child, CheckableComboBox):
                    if self.fc_type == "Accuri":
                        child.set_checked_items(self.attributes[child.name])
                elif isinstance(child, EditLine):
                    child.setText(str(self.attributes[child.name]))

    def update_config(self) -> None:
        """
        Updates the values of attributes after making changes in the settings window, and saves them in the
        configuration file.
        Returns:
            None
        """
        for widget in [self.widget_general, self.widget_vis, self.widget_nn, self.widget_data]:
            for child in widget.children():
                if isinstance(child, ComboBox):
                    self.attributes[child.name] = child.currentText()
                elif isinstance(child, CheckableComboBox):
                    self.attributes[child.name] = child.get_check_items()
                elif isinstance(child, EditLine):
                    self.attributes[child.name] = int(child.text())
                elif isinstance(child, Label) and child.name:
                    try:
                        self.attributes[child.name] = str(child.text())
                    except AttributeError:
                        pass
        self.settings.set_attributes(self.attributes)
        self.settings.save_settings()
