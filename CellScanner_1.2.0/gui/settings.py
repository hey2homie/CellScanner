from utilities.settings import Settings, SettingsOptions

from gui.widgets import Widget, Button, HLine, ComboBox, CheckableComboBox, EditLine, Label, TextEdit


class SettingsWindow(Widget):

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            stack (QStackedWidget): The stacked widget containing the all application's windows.
            settings (Settings): The settings class.
            *args: Optional arguments of the QWidget class.
            **kwargs: Optional arguments of the QWidget class.
        """
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
        ComboBox(obj_name="combobox", geometry=[201, 148, 108, 30], name="fc_type", parent=self)
        Label(text="Hardware:", obj_name="small", geometry=[332, 148, 88, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[427, 148, 85, 30], name="hardware", parent=self)
        Label(text="Output location:", obj_name="small", geometry=[46, 185, 146, 30], parent=self)
        Label(obj_name="small", geometry=[201, 185, 526, 30], name="results", parent=self)
        Button(text="Change", obj_name="settings", geometry=[736, 185, 65, 30], parent=self)
        Label(text="Visualizations", obj_name="settings", geometry=[46, 234, 157, 41], parent=self)
        HLine(obj_name="line", geometry=[204, 254, 643, 5], parent=self)
        Label(text="Visualisation type:", obj_name="small", geometry=[46, 289, 146, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[201, 289, 110, 30], name="vis_type", parent=self)
        Label(text="Dimension:", obj_name="small", geometry=[334, 289, 98, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[442, 289, 50, 30], name="vis_dims", parent=self)
        Label(text="Number of cores to compute UMAP:", obj_name="small", geometry=[520, 289, 275, 30], parent=self)
        EditLine(obj_name="input", geometry=[804, 289, 25, 30], name="num_umap_cores", parent=self)
        # TODO: Two lines below — swap number of cores with channels to use
        # Label(text="Channels to use:", obj_name="small", geometry=[28, 70, 123, 30], parent=self)
        # CheckableComboBox(obj_name="combobox", geometry=[130, 70, 90, 30], name="vis_channels", parent=self)
        Label(text="Neural Networks", obj_name="settings", geometry=[46, 339, 197, 41], parent=self)
        HLine(obj_name="line", geometry=[244, 361, 603, 5], parent=self)
        Label(text="Classifier:", obj_name="small", geometry=[46, 394, 146, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[198, 394, 224, 30], name="model", parent=self)
        Button(text="?", obj_name="settings", geometry=[432, 394, 25, 30], name="classifier", parent=self)
        Label(text="Batch size:", obj_name="small", geometry=[479, 394, 146, 30], parent=self)
        EditLine(obj_name="input", geometry=[634, 394, 46, 30], name="num_batches", parent=self)
        Label(text="Epochs:", obj_name="small", geometry=[703, 394, 69, 30], parent=self)
        EditLine(obj_name="input", geometry=[781, 394, 39, 30], name="num_epochs", parent=self)
        Label(text="Learning rate scheduler:", obj_name="small", geometry=[46, 432, 185, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[238, 432, 180, 30], name="lr_scheduler", parent=self)
        Label(text="Initial learning rate:", obj_name="small", geometry=[441, 432, 146, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[606, 432, 85, 30], name="lr", parent=self)
        Label(text="Data Preparation", obj_name="settings", geometry=[46, 481, 197, 41], parent=self)
        HLine(obj_name="line", geometry=[244, 502, 705, 5], parent=self)
        Label(text="Gating type:", obj_name="small", geometry=[46, 537, 146, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[201, 537, 150, 30], name="gating_type", parent=self)
        Label(text="Autoencoder:", obj_name="small", geometry=[46, 576, 146, 30], parent=self)
        ComboBox(obj_name="combobox", geometry=[201, 576, 236, 30], name="autoencoder", parent=self)
        Button(text="?", obj_name="settings", geometry=[442, 576, 25, 30], name="ae", parent=self)
        Label(text="Reconstruction Error:", obj_name="small", geometry=[491, 576, 164, 30], parent=self)
        EditLine(obj_name="input", geometry=[664, 576, 42, 30], name="mse_threshold", parent=self)
        Button(text="Menu", obj_name="standard", geometry=[46, 618, 200, 60], parent=self)
        Button(text="Apply", obj_name="standard", geometry=[652, 618, 200, 60], parent=self)
        """Label(text=self.model_info.get_readable(model="classifier", name=self.settings.model), obj_name="overlay",
              name="classifier", geometry=[370, 104, 300, 100], parent=self)
        Label(text=self.model_info.get_readable(model="ae", name=self.settings.autoencoder), obj_name="overlay",
              name="ae", geometry=[379, 306, 300, 100], parent=self)"""
        TextEdit(obj_name="overlay", name="classifier", geometry=[370, 104, 300, 100], parent=self)
        TextEdit(obj_name="overlay", name="ae", geometry=[379, 306, 300, 100], parent=self)
        self.children()[-1].hide()
        self.children()[-1].setText(self.model_info.get_readable(model="ae", name=self.settings.autoencoder))
        # self.children()[-1].setWordWrap(True)
        self.children()[-2].hide()
        self.children()[-2].setText(self.model_info.get_readable(model="classifier", name=self.settings.model))
        # self.children()[-2].setWordWrap(True)
        self.children()[-8].currentIndexChanged.connect(lambda: self.update_layouts(8))
        self.children()[-23].currentIndexChanged.connect(lambda: self.update_layouts(23))

    def update_layouts(self, combobox: int):
        name = self.children()[-combobox].currentText()
        try:
            if combobox == 8:
                self.children()[-1].setText(self.model_info.get_readable(model="ae", name=name))
            else:
                self.children()[-2].setText(self.model_info.get_readable(model="classifier", name=name))
        except KeyError:
            pass

    def set_values_from_config(self) -> None:
        for child in self.children():
            if isinstance(child, (ComboBox, CheckableComboBox)):
                child.clear()
                if child.name != "vis_channels" and child.name != "cols_to_drop":
                    child.addItems(self.combo_boxes_content[child.name])
                else:
                    if self.settings.fc_type == "Accuri":
                        child.addItems(self.combo_boxes_content["vis_channels_accuri"])
                    else:
                        child.addItems(self.combo_boxes_content["vis_channels_cytoflex"])
                if isinstance(child, ComboBox):
                    items = self.combo_boxes_content[child.name]
                    try:
                        child.setCurrentIndex(items.index(getattr(self.settings, child.name)))
                    except ValueError:
                        child.setCurrentIndex(items.index(str(getattr(self.settings, child.name))))
                elif isinstance(child, CheckableComboBox):
                    fc = "_accuri" if self.settings.fc_type == "Accuri" else "_cytoflex"
                    child.set_checked_items(getattr(self.settings, child.name + fc))
                    setattr(self.settings, child.name + fc, child.get_check_items())
            elif isinstance(child, (Label, EditLine)):
                try:
                    child.setText(str(self.settings.__dict__[child.name]))
                except KeyError:
                    pass

    def update_config(self) -> None:
        for child in self.children():
            if isinstance(child, ComboBox):
                setattr(self.settings, child.name, child.currentText())
            elif isinstance(child, CheckableComboBox):
                if child.name != "vis_channels" or child.name != "cols_to_drop":
                    setattr(self.settings, child.name, child.get_check_items())
                else:
                    fc = "_accuri" if self.settings.fc_type == "Accuri" else "_cytoflex"
                    setattr(self.settings, child.name + fc, child.get_check_items())
            elif isinstance(child, EditLine):
                setattr(self.settings, child.name, int(child.text()))
            elif isinstance(child, Label) and child.name:
                try:
                    setattr(self.settings, child.name, str(child.text()))
                except AttributeError:
                    pass
        self.settings.save_settings()
