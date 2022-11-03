import os
import sys

from PyQt6.QtWidgets import QApplication

from gui.start_window import MainWindow
from gui.file_selection import FileSelector
from gui.results import ResultsWindow
from gui.settings import SettingsWindow
from gui.about import AboutWindow
from gui.widgets import Stack


def main():
    app = QApplication(sys.argv)
    stack = Stack()
    start = MainWindow(stack=stack)
    FileSelector(stack=stack)
    ResultsWindow(stack=stack, settings=start.get_settings())
    SettingsWindow(stack=stack, settings=start.get_settings())
    AboutWindow(stack=stack)
    stack.setCurrentIndex(0)
    stack.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    os.chdir("./")
    main()
