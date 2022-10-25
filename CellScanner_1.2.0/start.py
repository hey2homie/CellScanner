import os
import sys

from PyQt6.QtWidgets import QApplication

from gui.start_window import MainWindow
from gui.file_selection import FileSelector
from gui.results import ResultsWindow
from gui.settings import SettingsWindow
from gui.about import AboutWindow


def main():
    app = QApplication(sys.argv)
    start = MainWindow()
    FileSelector(stack=start.get_stack())
    ResultsWindow(stack=start.get_stack(), settings=start.get_settings())
    SettingsWindow(stack=start.get_stack(), settings=start.get_settings())
    AboutWindow(stack=start.get_stack())
    sys.exit(app.exec())


if __name__ == "__main__":
    os.chdir("./")
    main()
