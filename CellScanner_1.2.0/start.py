import os
import sys

from PyQt6.QtWidgets import QApplication

from gui.start_window import MainWindow
from gui.file_selection import FileSelector
from gui.results import ResultsClassification, ResultsTraining
from gui.settings import SettingsWindow
from gui.about import AboutWindow
from gui.widgets import Stack


def main():
    app = QApplication(sys.argv)
    stack = Stack()
    MainWindow(stack=stack)
    FileSelector(stack=stack)
    ResultsClassification(stack=stack)
    ResultsTraining(stack=stack)
    SettingsWindow(stack=stack)
    AboutWindow(stack=stack)
    stack.setCurrentIndex(0)
    stack.center()
    stack.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    os.chdir("./")
    main()
