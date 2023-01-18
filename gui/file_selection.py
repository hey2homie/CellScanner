from utilities.classification_utils import ClassificationModel, AutoEncoder

from .widgets import Widget, DropBox, Button, EditLine, CheckBox, MessageBox
from utilities.helpers import split_files


class FileSelector(Widget):
    """
    Attributes:
    ----------
    action: str
        Action to be performed after adding files. Can be either "Training", "Prediction", or "Tool\nDiagnostics"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = None

    def __init_ui(self):
        """
        Initialize widgets. Depending on the action, different widgets are added like check-box for autoencoder or field
        for the model name.
        Returns:
            None.
        """
        self.setWindowTitle("File Selection")
        DropBox(obj_name="drop", geometry=[46, 43, 808, 450], parent=self)
        Button(
            text="Menu", obj_name="standard", geometry=[46, 518, 200, 60], parent=self
        )
        Button(
            text="Clear Data",
            obj_name="standard",
            geometry=[349, 518, 200, 60],
            parent=self,
        )
        Button(
            text="Predict",
            obj_name="standard",
            geometry=[652, 518, 200, 60],
            parent=self,
        )
        if self.action == "Tool\nDiagnostics":
            self.children()[3].setText("Diagnose")
        elif self.action == "Training":
            model_name = EditLine(
                obj_name="input", geometry=[46, 43, 638, 30], parent=self
            )
            model_name.setPlaceholderText("Enter model name")
            CheckBox(
                text="Autoencoder",
                obj_name="use_ae",
                geometry=[703, 43, 150, 30],
                parent=self,
            )
            self.children()[0].setGeometry(46, 93, 808, 400)
            self.children()[3].setText("Train")

    def set_action(self, action: str) -> None:
        """
        Set action to be performed after adding files. Reset widgets if action is changed.
        Args:
            action (str): Action to be performed after adding files. Can be "Prediction", "Training".
        Returns:
            None.
        """
        self.action = action
        [children.setParent(None) for children in self.children()]
        self.__init_ui()

    def run_action(self) -> None:
        """
        Depending on what button was pressed, run the appropriate action after adding files.
        Returns:
            None.
        """
        files = self.children()[0].get_files()
        if self.children()[0].count() != 0:
            settings = self.stack.widget(0).settings
            models_info = self.stack.widget(0).models_info
            if self.action == "Training":
                model_name = self.children()[4].text()
                if model_name == "":
                    MessageBox.about(self, "Warning", "Model name is not provided")
                    return
                model_name = model_name + ".h5"
                if self.children()[5].isChecked():
                    model = AutoEncoder(
                        settings=settings,
                        model_info=models_info,
                        files=files,
                        model_type="ae",
                        name=model_name,
                        training_cls=False,
                    )
                else:
                    model = ClassificationModel(
                        settings=settings,
                        model_info=models_info,
                        files=files,
                        model_type="classifier",
                        name=model_name,
                        training_cls=True,
                    )
                model.run_training()
                self.stack.widget(4).set_values_from_config()
                self.stack.widget(3).run_tf_board(name=model_name)
                self.stack.setCurrentIndex(3)
                self.stack.currentWidget().center()
                self.stack.setGeometry(*[300, 300, 1080, 700])
                self.stack.center()
            else:
                if settings.model == "":
                    MessageBox.about(self, "Warning", "Model is not provided")
                    return
                model = ClassificationModel(
                    settings=settings,
                    model_info=models_info,
                    files=files,
                    model_type="classifier",
                    name=settings.model,
                )
                if self.action == "Tool\nDiagnostics":
                    outputs = model.run_diagnostics()
                elif self.action == "Prediction":
                    outputs = model.run_classification()
                    files, _ = split_files(files, settings.gating_type)
                    self.stack.widget(2).set_items(files)
                self.stack.widget(2).inputs = outputs
                self.stack.widget(2).set_inputs()
                self.stack.setCurrentIndex(2)
        else:
            MessageBox.about(self, "Warning", "Files are not selected")
