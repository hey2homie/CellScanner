from PyQt6.QtWidgets import QWidget, QStackedWidget
from PyQt6.QtCore import Qt


class AboutWindow(QWidget):

    def __init__(self, stack: QStackedWidget, *args, **kwargs):
        super(AboutWindow, self).__init__(*args, **kwargs)
        self.stack = stack
        self.__init_ui()

    def __init_ui(self) -> None:
        self.setWindowTitle("About")
        self.setGeometry(self.stack.currentWidget().geometry())
        self.__init_widget()
        self.stack.addWidget(self)

    def __init_widget(self) -> None:
        from widgets import Label, TextEdit, Button
        tittle = Label(text="About CellScanner", obj_name="tittle", geometry=[0, 9, 895, 69], parent=self)
        tittle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        manual = TextEdit(obj_name="markdown", geometry=[46, 91, 808, 402], parent=self)
        manual.setMarkdown(self.__get_markdown())
        manual.setReadOnly(True)
        Button(text="Menu", obj_name="standard", geometry=[46, 518, 200, 60], parent=self)

    @staticmethod
    def __get_markdown() -> str:
        with open("gui/markdown_files/about.md") as md:
            return md.read()

