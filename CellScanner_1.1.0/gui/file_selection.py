from PyQt6.QtWidgets import QWidget, QFileDialog
from visualizations import Scatter3D

class FileSelection(QWidget):

    def __init__(self):
        super().__init__()
        self.title = "Choose File"
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.__init_ui()
        self.files = None

    def __init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.__open_file_names_dialog()
        self.show()

    def __open_file_names_dialog(self):
        options = QFileDialog.Option.DontUseNativeDialog
        self.files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                     "All Files (*);;Python Files (*.py)", options=options)
        Scatter3D().show()

    def get_files(self) -> list:
        return self.files
