from PyQt6.QtCore import Qt

from gui.widgets import Widget, Label, TextEdit, Button


class AboutWindow(Widget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__init_ui()

    def __init_ui(self) -> None:
        self.setWindowTitle("About")
        self.__init_widget()

    def __init_widget(self) -> None:
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

