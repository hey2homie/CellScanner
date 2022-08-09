from PyQt6.QtWidgets import QWidget, QStackedWidget


class FileSelector(QWidget):

    def __init__(self, stack: QStackedWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.title = "Choose Files"
        self.diagnostics = False
        self.__init_ui()

    def __init_ui(self):
        from widgets import DropBox, Button
        self.setWindowTitle(self.title)
        self.setGeometry(self.stack.currentWidget().geometry())
        DropBox(obj_name="drop", geometry=[46, 43, 808, 450], parent=self)
        Button(text="Menu", obj_name="standard", geometry=[46, 518, 200, 60], parent=self)
        Button(text="Clear Data", obj_name="standard", geometry=[349, 518, 200, 60], parent=self)
        Button(text="Next", obj_name="standard", geometry=[652, 518, 200, 60], parent=self)
        self.stack.addWidget(self)
