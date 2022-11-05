from gui.widgets import Widget, Label, Button
from utilities.settings import Settings, ModelsInfo


class MainWindow(Widget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = Settings()
        self.models_info = ModelsInfo()
        self.__init_ui()

    def __init_ui(self) -> None:
        self.__window_params()
        self.setWindowTitle("CellScanner")
        self.__init_widgets()

    def __window_params(self) -> None:
        self.setGeometry(300, 300, 895, 600)
        self.stack.setGeometry(self.geometry())

    def __init_widgets(self) -> None:
        Label(text="CellScanner", obj_name="tittle", geometry=[0, 9, 895, 69], parent=self)
        Label(text="Version 1.2\nC. Joseph", obj_name="version", geometry=[0, 531, 895, 58], parent=self)
        Button(text="Prediction", obj_name="main", geometry=[27, 91, 166, 425], parent=self)
        Button(text="Training", obj_name="main", geometry=[197, 91, 166, 425], parent=self)
        Button(text="Tool\nDiagnostics", obj_name="main", geometry=[367, 91, 166, 425], parent=self)
        Button(text="Settings", obj_name="main", geometry=[537, 91, 166, 425], parent=self)
        Button(text="Help", obj_name="main", geometry=[707, 91, 166, 425], parent=self)

    def get_settings(self):
        return self.settings

    def get_settings(self):
        return self.settings

    def get_models_info(self):
        return self.models_info
