from PyQt6.QtWidgets import QMainWindow, QStackedWidget
from PyQt6.QtCore import Qt

from gui.widgets import Label, Button
from utilities.settings import Settings, ModelsInfo


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.settings = Settings()
        self.models_info = ModelsInfo()
        self.stack = QStackedWidget()
        self.stack.addWidget(self)
        self.__init_ui()

    def __init_ui(self) -> None:
        self.__window_params()
        self.setWindowTitle("CellScanner")
        self.stack.setGeometry(self.geometry())
        self.stack.setCurrentIndex(0)
        self.__init_widgets()
        self.stack.show()

    def __init_widgets(self) -> None:
        tittle = Label(text="CellScanner", obj_name="tittle", geometry=[0, 9, 895, 69], parent=self)
        version = Label(text="Version 1.2\nC. Joseph", obj_name="version", geometry=[0, 531, 895, 58], parent=self)
        [label.setAlignment(Qt.AlignmentFlag.AlignCenter) for label in [tittle, version]]
        Button(text="Prediction", obj_name="main", geometry=[27, 91, 166, 425], parent=self)
        Button(text="Training", obj_name="main", geometry=[197, 91, 166, 425], parent=self)
        Button(text="Tool\nDiagnostics", obj_name="main", geometry=[367, 91, 166, 425], parent=self)
        Button(text="Settings", obj_name="main", geometry=[537, 91, 166, 425], parent=self)
        Button(text="Help", obj_name="main", geometry=[707, 91, 166, 425], parent=self)

    def __window_params(self) -> None:
        self.setGeometry(300, 300, 895, 600)
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # TODO: Pythonic way for getters
    def get_settings(self) -> Settings:
        return self.settings

    def set_settings(self) -> None:
        self.settings.save_settings()

    def get_models_info(self) -> ModelsInfo:
        return self.models_info

    def get_stack(self) -> QStackedWidget:
        return self.stack
