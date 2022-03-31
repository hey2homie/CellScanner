import sys
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtGui import QAction, QIcon, QPixmap
from file_selection import FileSelection


class Main(QMainWindow):

    def __init__(self, app: QApplication, *args, **kwargs):
        super(Main, self).__init__(*args, **kwargs)
        self.app = app
        self.__init_toolbar()
        self.__init_ui()

    @staticmethod
    def __labels() -> list:
        title = QLabel("CellScanner")
        logo = QLabel()
        logo_img = QPixmap("../icons/logo.png")
        logo_img = logo_img.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.FastTransformation)
        logo.setPixmap(logo_img)
        logo.resize(logo_img.width(), logo_img.height())
        message = QLabel("Let's predict what is \n in your medium!")
        version = QLabel("Version 1.1.0\nC. JOSEPH")
        return [title, logo, message, version]

    @staticmethod
    def __buttons() -> list:
        prediction = QPushButton("New Prediction")
        tool_analysis = QPushButton("Tool Analysis")
        clustering = QPushButton("Clustering")
        clustering_analysis = QPushButton("Clustering Analysis")
        unsupervised_clst = QPushButton("Unsupervised Clustering")
        upd_data = QPushButton("Update Data")
        return [prediction, tool_analysis, clustering, clustering_analysis, unsupervised_clst, upd_data]

    @staticmethod
    def __button_funcs() -> tuple:
        # TODO: Change classes later
        prediction = partial(lambda: FileSelection().show())
        tool = partial(lambda: FileSelection().show())
        clustering = partial(lambda: FileSelection().show())
        clustering_analysis = partial(lambda: FileSelection().show())
        unsupervised_clst = partial(lambda: FileSelection().show())
        return prediction, tool, clustering, clustering_analysis, unsupervised_clst

    @staticmethod
    def __tool_bar_labels() -> tuple:
        return "Parameters", "Flow Cytometer", "Help", "Exit"

    def __tool_bar_actions(self) -> list:
        param = QAction(QIcon("../icons/param.png"), "Parameters", self)
        fc = QAction(QIcon("../icons/fc.png"), "Flow Cytometer", self)
        guide = QAction(QIcon("../icons/help.png"), "Help", self)
        exit_app = QAction(QIcon("../icons/exit.png"), "Exit", self)
        return [param, fc, guide, exit_app]

    def __tool_bar_functions(self) -> tuple:
        param = partial(lambda: sys.exit(self.app.exec()))
        fc = partial(lambda: sys.exit(self.app.exec()))
        guide = partial(lambda: sys.exit(self.app.exec()))
        exit_app = partial(lambda: sys.exit(self.app.exec()))
        return param, fc, guide, exit_app

    def __init_ui(self) -> None:
        self.__window_params()
        layout = QVBoxLayout()
        labels = self.__labels()
        buttons = self.__buttons()
        button_func = self.__button_funcs()
        for function in range(len(button_func)):
            buttons[function].clicked.connect(button_func[function])
        [label.setAlignment(Qt.AlignmentFlag.AlignCenter) for label in labels]
        [layout.addWidget(label) for label in labels[:-1] + buttons]
        layout.addWidget(labels[-1])
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def __window_params(self) -> None:
        self.setGeometry(300, 300, 300, 200)
        self.resize(250, 450)
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def __init_toolbar(self) -> None:
        tool_bar_labels = self.__tool_bar_labels()
        tool_bar_actions = self.__tool_bar_actions()
        tool_bar_func = self.__tool_bar_functions()
        for elem in range(len(tool_bar_labels)):
            tool_bar_actions[elem].triggered.connect(tool_bar_func[elem])
            self.tool_bar = self.addToolBar(tool_bar_labels[elem])
            self.tool_bar.addAction(tool_bar_actions[elem])


def main():
    app = QApplication(sys.argv)
    window = Main(app)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
