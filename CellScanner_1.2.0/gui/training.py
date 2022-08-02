from PyQt6.QtWidgets import QWidget, QStackedWidget, QMessageBox

from utilities.classification_utils import ClassificationTraining
from utilities.settings import Settings


class TrainingWindow(QWidget):

    def __init__(self, stack: QStackedWidget, settings: Settings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.settings = settings
        self.title = "Choose Files"
        self.__init_ui()

    def __init_ui(self):
        from widgets import DropBox, Button, EditLine
        self.setWindowTitle(self.title)
        self.setGeometry(self.stack.currentWidget().geometry())
        model_name = EditLine(obj_name="input", geometry=[46, 43, 808, 30], parent=self)
        model_name.setPlaceholderText("Enter model name")
        DropBox(obj_name="drop", geometry=[46, 93, 808, 400], parent=self)
        Button(text="Menu", obj_name="standard", geometry=[46, 518, 200, 60], parent=self)
        Button(text="Clear Data", obj_name="standard", geometry=[349, 518, 200, 60], parent=self)
        Button(text="Train", obj_name="standard", geometry=[652, 518, 200, 60], parent=self)
        self.stack.addWidget(self)

    def begin_training(self) -> None:
        model_name = self.children()[0].text()
        if model_name == "":
            QMessageBox.about(self, "Warning", "Model name is not provided")
            return
        if self.children()[1].count() != 0:
            files = self.children()[1].get_files()
            models_info = self.stack.widget(0).get_models_info()
            training = ClassificationTraining(files=files, model_name=model_name, settings=self.settings,
                                              models_info=models_info)
            training.run_training()
        else:
            QMessageBox.about(self, "Warning", "Files are not selected")
