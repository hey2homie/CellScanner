import os
from glob import glob

from PyQt6.QtCore import Qt

from utilities.settings import SettingsOptions
from utilities.helpers import get_available_models_fc, get_available_cls

from gui.widgets import Widget, Button, HLine, ComboBox, CheckableComboBox, EditLine, Label, TextEdit, CheckBox


class SettingsWindow(Widget):
    """
    Attributes:
    ----------
    settings: Settings
        Settings object, containing the settings for the current run.
    models_info: ModelsInfo
        ModelsInfo object, containing the model's metadata.
    combo_boxes_content: dict
        A dictionary containing the content for the combo-boxes taken from SettingsOptions enum.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.settings = self.stack.widget(0).settings
        self.model_info = self.stack.widget(0).models_info
        self.combo_boxes_content = SettingsOptions.get_dictionary()
        self.__init_ui()

    def __init_ui(self) -> None:
        self.setWindowTitle("Settings")
        self.__init_widgets()
        self.set_values_from_config()

    def __init_widgets(self) -> None:
        Label(text="Settings", obj_name="tittle", geometry=[46, 9, 803, 69], parent=self)
        Label(text="General", obj_name="settings", geometry=[46, 93, 98, 41], parent=self)
        HLine(obj_name="line", geometry=[146, 113, 701, 5], parent=self)
        Label(text="FC type:", obj_name="small", geometry=[46, 148, 146, 30], parent=self)
        fc = ComboBox(obj_name="combobox", geometry=[201, 148, 108, 30], name="fc_type", parent=self)
        Label(text="Hardware:", obj_name="small", geometry=[332, 148, 88, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[427, 148, 85, 30], name="hardware", parent=self)
        Label(text="Output location:", obj_name="small", geometry=[46, 185, 146, 30], parent=self)
        Label(obj_name="small", geometry=[201, 185, 526, 30], name="results", parent=self)
        Button(text="Change", obj_name="settings", geometry=[736, 185, 65, 30], parent=self)
        Label(text="Visualizations", obj_name="settings", geometry=[46, 234, 157, 41], parent=self)
        HLine(obj_name="line", geometry=[204, 254, 643, 5], parent=self)
        Label(text="Visualisation type:", obj_name="small", geometry=[46, 289, 146, 30], parent=self)
        vis_type = ComboBox(obj_name="combobox", geometry=[201, 289, 110, 30], name="vis_type", parent=self)
        Label(text="Dimension:", obj_name="small", geometry=[334, 289, 98, 30], parent=self)
        vis_dims = ComboBox(obj_name="combobox", geometry=[442, 289, 55, 30], name="vis_dims", parent=self)
        cores = Label(text="Number of cores to compute UMAP:", obj_name="small", geometry=[525, 289, 275, 30],
                      parent=self)
        cores_input = EditLine(obj_name="input", geometry=[809, 289, 25, 30], name="num_umap_cores", parent=self)
        channels = Label(text="Channels to use:", obj_name="small", geometry=[525, 289, 131, 30], parent=self)
        channels_input = CheckableComboBox(obj_name="combobox", geometry=[664, 289, 142, 30], name="vis_channels",
                                           parent=self)
        Label(text="Neural Networks", obj_name="settings", geometry=[46, 339, 197, 41], parent=self)
        HLine(obj_name="line", geometry=[244, 361, 603, 5], parent=self)
        Label(text="Classifier:", obj_name="small", geometry=[46, 394, 146, 30], parent=self)
        classifiers = ComboBox(obj_name="combobox", geometry=[198, 394, 224, 30], name="model", parent=self)
        Button(text="?", obj_name="settings", geometry=[432, 394, 25, 30], name="classifier", parent=self)
        Label(text="Batch size:", obj_name="small", geometry=[479, 394, 146, 30], parent=self)
        EditLine(obj_name="input", geometry=[634, 394, 46, 30], name="num_batches", parent=self)
        Label(text="Epochs:", obj_name="small", geometry=[703, 394, 69, 30], parent=self)
        EditLine(obj_name="input", geometry=[781, 394, 39, 30], name="num_epochs", parent=self)
        Label(text="Learning rate scheduler:", obj_name="small", geometry=[46, 432, 185, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[238, 432, 180, 30], name="lr_scheduler", parent=self)
        Label(text="Initial learning rate:", obj_name="small", geometry=[441, 432, 146, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[606, 432, 85, 30], name="lr", parent=self)
        CheckBox(text="Legacy NN", obj_name="mlp", geometry=[713, 432, 104, 30], parent=self)
        Label(text="Data Preparation", obj_name="settings", geometry=[46, 481, 197, 41], parent=self)
        HLine(obj_name="line", geometry=[244, 502, 603, 5], parent=self)
        Label(text="Gating type:", obj_name="small", geometry=[46, 537, 146, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[201, 537, 150, 30], name="gating_type", parent=self)
        Label(text="Channels to drop:", obj_name="small", geometry=[374, 537, 141, 30], parent=self)
        cols_drop = CheckableComboBox(obj_name="combobox", geometry=[523, 537, 142, 30], name="cols_to_drop",
                                      parent=self)
        Label(text="Autoencoder:", obj_name="small", geometry=[46, 576, 146, 30], parent=self)
        aes = ComboBox(obj_name="combobox", geometry=[201, 576, 236, 30], name="autoencoder", parent=self)
        Button(text="?", obj_name="settings", geometry=[442, 576, 25, 30], name="ae", parent=self)
        Label(text="Reconstruction Error:", obj_name="small", geometry=[491, 576, 164, 30], parent=self)
        EditLine(obj_name="input", geometry=[664, 576, 42, 30], name="mse_threshold", parent=self)
        Button(text="Menu", obj_name="standard", geometry=[46, 618, 200, 60], parent=self)
        Button(text="Apply", obj_name="standard", geometry=[652, 618, 200, 60], parent=self)
        overlay_classifier = TextEdit(obj_name="overlay", name="classifier", geometry=[370, 104, 300, 100], parent=self)
        overlay_ae = TextEdit(obj_name="overlay", name="ae", geometry=[379, 306, 300, 100], parent=self)

        overlay_ae.hide()
        overlay_ae.setText(self.model_info.get_readable(model="ae", name=self.settings.autoencoder))
        overlay_classifier.hide()
        overlay_classifier.setText(self.model_info.get_readable(model="classifier", name=self.settings.model))

        fc.currentIndexChanged.connect(lambda: self.__update_fc_related(fc, channels_input, cols_drop,
                                                                        aes, classifiers))
        vis_type.currentIndexChanged.connect(lambda: self.__update_vis(vis_type, cores, cores_input, channels,
                                                                       channels_input))
        vis_dims.currentIndexChanged.connect(channels_input.item_unchecked)
        aes.currentIndexChanged.connect(lambda: self.__update_layouts(aes, overlay_ae, channels_input, classifiers))
        classifiers.currentIndexChanged.connect(lambda: self.__update_layouts(classifiers, overlay_classifier))

    def __update_fc_related(self, combobox: ComboBox, vis: CheckableComboBox, drop: CheckableComboBox, ae: ComboBox,
                            cls: ComboBox) -> None:
        """
        Updates the channels to use and the channels to drop combo-boxes when the fc type is changed. Additionally,
        updates the autoencoder and classifier combo-boxes.
        Args:
            combobox (ComboBox): The combo-box to be updated.
            vis (CheckableComboBox): The channels to be updated.
            drop (CheckableComboBox): The columns to be updated.
            ae (ComboBox): The autoencoders list to be updated.
            cls (ComboBox): The classifiers list to be updated.
        Returns:
            None.
        """
        fc = "_" + combobox.currentText().lower()
        if fc == "_":
            fc = fc + self.settings.fc_type.lower()
        for widget in [vis, drop]:
            widget.clear()
            widget.addItems(self.combo_boxes_content[widget.name + fc])
            widget.set_checked_items(getattr(self.settings, widget.name + fc))
        widgets = {cls: self.model_info.classifiers, ae: self.model_info.autoencoders}
        for widget, data in widgets.items():
            widget.clear()
            widget.addItems(get_available_models_fc(data, fc))

    def __update_vis(self, vis_type: ComboBox, cores: Label, cores_input: EditLine, channels: Label,
                     channels_input: CheckableComboBox) -> None:
        """
        Updates the number of cores and the channels to use combo-boxes when the vis type is changed.
        Args:
            vis_type (ComboBox): The combo-box to be updated.
            cores (Label): The label to be updated.
            cores_input (EditLine): The edit line to be updated.
            channels (Label): The label to be updated.
            channels_input (CheckableComboBox): The combo-box to be updated.
        Returns:
            None.
        """
        if vis_type.currentText() is None:
            vis_type = self.settings.vis_type
        else:
            vis_type = vis_type.currentText()
        if vis_type == "UMAP":
            cores.show()
            cores_input.show()
            channels.hide()
            channels_input.hide()
        else:
            cores.hide()
            cores_input.hide()
            channels.show()
            channels_input.show()

    def __update_layouts(self, combobox: ComboBox, text: TextEdit, vis_channels: CheckableComboBox = None,
                         classifiers: ComboBox = None) -> None:
        """
        Updates the text in the overlay when the model is changed. Also, updated channels that are available for the
        visualization if the model is an autoencoder.
        Args:
            combobox (ComboBox): The combo-box to be updated.
            text (TextEdit): The text to be updated.
            vis_channels (CheckableComboBox, optional): The channels to be updated.
            classifiers (ComboBox): The classifiers list to be updated.
        Returns:
            None.
        """
        model_name = combobox.currentText()
        if model_name != "":
            features = self.model_info.get_readable(model=combobox.name, name=model_name)
            text.setText(features)
            if combobox.name == "autoencoder":
                vis_channels.clear()
                vis_channels.addItems(features.split(", "))
                fc = "_accuri" if self.settings.fc_type == "Accuri" else "_cytoflex"
                vis_channels.set_checked_items(getattr(self.settings, "vis_channels" + fc))
                available_classifiers = get_available_cls(self.model_info.classifiers, model_name)
                classifiers.clear()
                classifiers.addItems(available_classifiers)

    def set_values_from_config(self) -> None:
        """
        Sets the values of the widgets from the configuration file.
        Returns:
            None.
        """
        self.combo_boxes_content["model"] = [os.path.basename(file) for file in glob("./classifiers/*.h5")]
        self.combo_boxes_content["autoencoder"] = [os.path.basename(file) for file in glob("./autoencoders/*.h5")]
        for child in self.children():
            if isinstance(child, (ComboBox, CheckableComboBox)):
                child.clear()
                if child.name not in ["vis_channels", "cols_to_drop", "model", "autoencoder"]:
                    child.addItems(self.combo_boxes_content[child.name])
                else:
                    fc = "_accuri" if self.settings.fc_type == "Accuri" else "_cytoflex"
                    if child.name == "model":
                        models = get_available_cls(self.model_info.classifiers, self.settings.autoencoder)
                        self.combo_boxes_content[child.name] = models
                        child.addItems(models)
                    elif child.name == "autoencoder":
                        models = get_available_models_fc(self.model_info.autoencoders, fc)
                        self.combo_boxes_content[child.name] = models
                        child.addItems(models)
                    elif child.name == "vis_channels":
                        features = self.model_info.get_readable(model="autoencoder", name=self.settings.autoencoder)
                        child.addItems(features.split(", "))
                    else:
                        child.addItems(self.combo_boxes_content[child.name + fc])
                if isinstance(child, ComboBox):
                    items = self.combo_boxes_content[child.name]
                    try:
                        child.setCurrentIndex(items.index(getattr(self.settings, child.name)))
                    except ValueError:
                        child.setCurrentIndex(items.index(str(getattr(self.settings, child.name))))
                elif isinstance(child, CheckableComboBox):
                    fc = "_accuri" if self.settings.fc_type == "Accuri" else "_cytoflex"
                    child.set_checked_items(getattr(self.settings, child.name + fc))
            elif isinstance(child, (Label, EditLine)):
                try:
                    child.setText(str(self.settings.__dict__[child.name]))
                except KeyError:
                    pass
            elif isinstance(child, CheckBox):
                if self.settings.mlp:
                    child.setCheckState(Qt.CheckState.Checked)
                else:
                    child.setCheckState(Qt.CheckState.Unchecked)

    def update_config(self) -> None:
        """
        Updates the configuration file with the values from the widgets.
        Returns:
            None.
        """
        for child in self.children():
            if isinstance(child, ComboBox):
                try:
                    setattr(self.settings, child.name, int(child.currentText()))
                except ValueError:
                    setattr(self.settings, child.name, child.currentText())
            elif isinstance(child, CheckableComboBox):
                if child.name != "vis_channels" and child.name != "cols_to_drop":
                    setattr(self.settings, child.name, child.get_check_items())
                else:
                    fc = "_accuri" if self.settings.fc_type == "Accuri" else "_cytoflex"
                    setattr(self.settings, child.name + fc, child.get_check_items())
            elif isinstance(child, EditLine):
                if child.name == "mse_threshold":
                    setattr(self.settings, child.name, float(child.text()))
                else:
                    setattr(self.settings, child.name, int(child.text()))
            elif isinstance(child, Label) and child.name:
                try:
                    setattr(self.settings, child.name, str(child.text()))
                except AttributeError:
                    pass
            elif isinstance(child, CheckBox):
                self.settings.mlp = child.isChecked()
        self.settings.save_settings()
