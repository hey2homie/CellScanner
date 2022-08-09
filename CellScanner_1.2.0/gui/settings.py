import os
from enum import Enum

from PyQt6.QtWidgets import QWidget, QStackedWidget
from PyQt6.QtCore import Qt

from utilities.settings import Settings


class ComboBoxContent(Enum):
    """
    Enum containing options for the Combo Boxes used in SettingsWindow.
    """
    FC = iter(["Accuri", "Cytoflex"])
    Classifiers = iter(os.listdir("../classifiers/"))
    VisType = iter(["UMAP", "Channels"])
    Dims = iter(["2D", "3D"])
    Channels = iter(["Blabl", "Bla-1", "Bla-2F"])  # TODO: Update later
    Hardware = iter(["GPU", "CPU"])
    Lr = iter(["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"])
    Lr_reduced = iter(["1e-3", "1e-2"])
    LrScheduler = iter(["Constant", "Time Decay", "Step Decay", "Exponential Decay"])
    Gating = iter(["Line", "Machine"])
    # TODO: Change calling iter everytime to some getter that will apply iter() everytime attribute is called


class SettingsWindow(QWidget):
    fc_type = None
    model = None
    results = None
    vis_type = None
    vis_channels = None
    vis_dims = None
    num_umap_cores = None
    hardware = None
    num_batches = None
    num_epochs = None
    lr = None
    lr_scheduler = None
    lr_reduced = None
    gating_type = None

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
        from widgets import Widget, Styles, Button
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
        from widgets import Button, Label, HLine, ComboBox, EditLine

        def index_changed(widget: ComboBox):
            if widget.currentIndex() != 0:
                self.widget_nn.children()[11].show()
                self.widget_nn.children()[12].show()
            else:
                self.widget_nn.children()[11].hide()
                self.widget_nn.children()[12].hide()

        tittle_label = Label(text="Settings", obj_name="tittle", geometry=[0, 9, 895, 69], parent=self)
        general_label = Label(text="General", obj_name="settings", geometry=[0, 0, 98, 41],
                              parent=self.widget_general)
        fc_label = Label(text="FC type:", obj_name="small", geometry=[28, 37, 70, 26], parent=self.widget_general)
        fc_choose = ComboBox(obj_name="combobox", geometry=[100, 37, 150, 26], name="fc_type",
                             parent=self.widget_general)
        fc_choose.addItems(ComboBoxContent.FC.value)
        classifier_label = Label(text="Classifier:", obj_name="small", geometry=[280, 37, 79, 26],
                                 parent=self.widget_general)
        classifier_choose = ComboBox(obj_name="combobox", geometry=[340, 37, 440, 26], name="model",
                                     parent=self.widget_general)
        classifier_choose.addItems(ComboBoxContent.Classifiers.value)
        # TODO: Add pop-up window with info about which bacteria classifier can predict (from models_info.yaml)
        output_label = Label(text="Output location:", obj_name="small", geometry=[28, 70, 124, 26],
                             parent=self.widget_general)
        output_location_label = Label(text=self.results, obj_name="small", geometry=[160, 70, 604, 26], name="results",
                                      parent=self.widget_general)
        output_location_button = Button(text="...", obj_name="standard", geometry=[730, 70, 50, 26],
                                        parent=self.widget_general)
        HLine(obj_name="line", geometry=[100, 16, 705, 1], parent=self.widget_general)
        vis_label = Label(text="Visualizations", obj_name="settings", geometry=[0, 0, 153, 41],
                          parent=self.widget_vis)
        vis_type_label = Label(text="Visualisation type:", obj_name="small", geometry=[28, 37, 138, 26],
                               parent=self.widget_vis)
        vis_type_choose = ComboBox(obj_name="combobox", geometry=[141, 37, 110, 26], name="vis_type",
                                   parent=self.widget_vis)
        vis_type_choose.addItems(ComboBoxContent.VisType.value)
        dim_label = Label(text="Dimension:", obj_name="small", geometry=[280, 37, 97, 26], parent=self.widget_vis)
        dim_choose = ComboBox(obj_name="combobox", geometry=[358, 37, 65, 26], name="vis_dims", parent=self.widget_vis)
        dim_choose.addItems(ComboBoxContent.Dims.value)
        channels_label = Label(text="Used channels:", obj_name="small", geometry=[28, 70, 123, 26],
                               parent=self.widget_vis)
        channels_choose = ComboBox(obj_name="combobox", geometry=[130, 70, 90, 26], name="vis_channels",
                                   parent=self.widget_vis)
        channels_choose.addItems(ComboBoxContent.Channels.value)
        umap_cores_label = Label(text="Number of cores to compute UMAP:", obj_name="small", geometry=[257, 70, 270, 26],
                                 parent=self.widget_vis)
        umap_cores_input = EditLine(obj_name="input", geometry=[480, 70, 30, 26], name="num_umap_cores",
                                    parent=self.widget_vis)
        HLine(obj_name="line", geometry=[155, 16, 705, 1], parent=self.widget_vis)
        nn_label = Label(text="Neural Network", obj_name="settings", geometry=[0, 0, 178, 41],
                         parent=self.widget_nn)
        hardware_label = Label(text="Hardware used for training:", obj_name="small", geometry=[28, 37, 203, 26],
                               parent=self.widget_nn)
        hardware_choose = ComboBox(obj_name="combobox", geometry=[199, 37, 75, 26], name="hardware",
                                   parent=self.widget_nn)
        hardware_choose.addItems(ComboBoxContent.Hardware.value)
        lr_label = Label(text="Learning rate:", obj_name="small", geometry=[291, 37, 108, 26], parent=self.widget_nn)
        lr_choose = ComboBox(obj_name="combobox", geometry=[384, 37, 85, 26], name="lr", parent=self.widget_nn)
        lr_choose.addItems(ComboBoxContent.Lr.value)
        lr_scheduler_label = Label(text="Learning rate scheduler:", obj_name="small", geometry=[494, 37, 182, 26],
                                   parent=self.widget_nn)
        lr_scheduler_choose = ComboBox(obj_name="combobox", geometry=[646, 37, 135, 26], name="lr_scheduler",
                                       parent=self.widget_nn)
        lr_scheduler_choose.addItems(ComboBoxContent.LrScheduler.value)
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
        lr_initial_choose.addItems(ComboBoxContent.Lr_reduced.value)
        if lr_scheduler_choose.currentIndex() == 0:
            lr_initial_label.hide()
            lr_initial_choose.hide()
        HLine(obj_name="line", geometry=[180, 16, 705, 1], parent=self.widget_nn)
        data_label = Label(text="Data Preparation", obj_name="settings", geometry=[0, 0, 192, 41],
                           parent=self.widget_data)
        gating_label = Label(text="Gating type:", obj_name="small", geometry=[28, 37, 96, 26], parent=self.widget_data)
        gating_choose = ComboBox(obj_name="combobox", geometry=[118, 37, 110, 26], name="gating_type",
                                 parent=self.widget_data)
        gating_choose.addItems(ComboBoxContent.Gating.value)
        HLine(obj_name="line", geometry=[194, 16, 705, 1], parent=self.widget_data)
        # TODO: Make global alignment through iteration over children at the class initiation
        lr_scheduler_choose.currentIndexChanged.connect(lambda: index_changed(lr_scheduler_choose))
        tittle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        [label.setAlignment(Qt.AlignmentFlag.AlignLeft) for label in [general_label, vis_label, nn_label, data_label]]

    def __set_values_from_config(self) -> None:
        from widgets import ComboBox, EditLine
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
                    child.setCurrentIndex(items.index(self.attributes[child.name]))
                elif isinstance(child, EditLine):
                    child.setText(str(self.attributes[child.name]))

    def update_config(self) -> None:
        """
        Updates the values of attributes after making changes in the settings window, and saves them in the
        configuration file.
        Returns:
            None
        """
        from widgets import ComboBox, EditLine, Label
        for widget in [self.widget_general, self.widget_vis, self.widget_nn, self.widget_data]:
            for child in widget.children():
                if isinstance(child, ComboBox):
                    self.attributes[child.name] = child.currentText()
                elif isinstance(child, EditLine):
                    self.attributes[child.name] = int(child.text())
                elif isinstance(child, Label) and child.name:
                    try:
                        self.attributes[child.name] = str(child.text())
                    except AttributeError:
                        pass
        self.settings.set_attributes(self.attributes)
        self.settings.save_settings()
