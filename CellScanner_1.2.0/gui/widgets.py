from enum import Enum
from typing import Any

from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QListWidget, QFileDialog, QStackedWidget, QMessageBox, \
    QFrame, QComboBox, QLineEdit, QStyleOptionComboBox, QStyle
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPalette

from utilities.classification_utils import ClassificationResults
from utilities.settings import ModelsInfo
# TODO: Add button with pop-up window with info about selected model. See above


class Styles(Enum):
    """
    Enum containing locations for the stylesheets for various widgets.
    """
    Widget = "gui/style_sheets/widgets.css"
    Label = "gui/style_sheets/labels.css"
    Line = "gui/style_sheets/line.css"
    EditLine = "gui/style_sheets/editline.css"
    Button = "gui/style_sheets/buttons.css"
    ComboBox = "gui/style_sheets/combobox.css"
    DropArea = "gui/style_sheets/lists.css"


class Widget(QWidget):
    """
    Widget class inherits from QWidget class. Additional methods are introduced that are inherited by its children.
    Particularly, method set_geometry and set_style.
    """
    name = None

    def __init__(self, widget: Styles, obj_name: str, geometry: list = None, name: str = None, *args, **kwargs) -> None:
        """
        Args:
            widget (Styles): one of the Styles attributes.
            obj_name (str): string used by the stylesheet to apply particular settings to the given widget.
            geometry (list, optional): list of 4 ints specifying x and y position relative to the parent widget, widget
            height and width.
            name (str, optional): name of the widget.
            *args: additional argument that is inherited from the parent class.
            **kwargs: additional arguments that are inherited from the parent class.
        """
        super(Widget, self).__init__(*args, **kwargs)
        self.setObjectName(obj_name)
        if geometry:
            self.set_geometry(geometry=geometry)
        self.set_style(css=widget.value)
        if name:
            self.name = name

    def set_geometry(self, geometry: list, ) -> None:
        """
        Sets geometry of the widget.
        Args:
            geometry (list): list of 4 ints specifying x and y position relative to the parent widget, and widget height
            and width.
        """
        self.setGeometry(*geometry)

    def set_style(self, css: Any) -> None:
        """
        Sets widget style.
        Args:
            css (Any): path to the stylesheet, which is Style attribute value.
        """
        with open(css, "r") as file:
            style = file.read()
            self.setStyleSheet(style)

    def set_name(self, name: str) -> None:
        self.name = name


class Label(QLabel, Widget):
    """
    Modified QLabel class, which inherits set_geometry and set_class from Widget parent class.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
            **kwargs:
        """
        super(Label, self).__init__(widget=Styles.Label, *args, **kwargs)


class HLine(QFrame, Widget):
    """
    Modified QFrame class, which inherits set_geometry and set_class from Widget parent class. HLine is a simple
    horizontal line.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
            **kwargs:
        """
        super(HLine, self).__init__(widget=Styles.Line, *args, **kwargs)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


class EditLine(QLineEdit, Widget):
    """
    Modified QLineEdit class, which inherits set_geometry and set_class from Widget parent class.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args:
            **kwargs:
        """
        super(EditLine, self).__init__(widget=Styles.EditLine, *args, **kwargs)


class Button(QPushButton, Widget):
    """
    Modified QPushButton class, which inherits set_geometry and set_class from Widget parent class. Additionally, button
    actions are specified here. The exact action depends on the text of the button.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
            **kwargs:
        """
        super(Button, self).__init__(widget=Styles.Button, *args, **kwargs)
        self.__add_action()

    def __add_action(self) -> None:
        """
        Adds actions to the buttons which is triggered upon click.
        """
        def get_top_level_parent(widget: QWidget) -> QStackedWidget:
            """
            Returns first parent of widget which has QStackedWidget as class attribute.
            Args:
                widget (QWidget): Child widget whose top level parent is to be returned.
            Returns:
                widget (QStackedWidget): Parent widget containing QStackedWidget.
            """
            try:
                return widget.stack
            except AttributeError:
                return get_top_level_parent(widget.parent())

        def show_file_selection_window(stack: QStackedWidget) -> None:
            """
            Shows file selection window with diagnostics button instead of standard.
            Args:
                stack (QStackedWidget): Stack of widgets.
            """
            stack.setCurrentIndex(1)
            if self.text() == "Tool\nDiagnostics":
                stack.widget(1).children()[3].setText("Diagnostics")
            else:
                stack.widget(1).children()[3].setText("Next")

        def clear(stack: QStackedWidget) -> None:
            """
            Clears the file input field, and switches window to the main window.
            Args:
                stack (QStackedWidget): Stack of widgets.
            """
            if stack.currentIndex() != 3:
                stack.widget(1).children()[0].clear()
                stack.widget(2).children()[1].clear()
                stack.widget(4).children()[1].clear()
            stack.setCurrentIndex(0)

        def show_results(stack: QStackedWidget) -> None:
            """
            Runs classification or diagnostics on provided files and displays results.
            Args:
                stack (QStackedWidget): stack of widgets.
            """
            if stack.widget(1).children()[0].count() != 0:
                files = stack.widget(1).children()[0].get_files()
                settings = stack.widget(0).get_settings()
                model_settings = stack.widget(0).get_models_info()
                model_settings.classifier_name = settings.model
                # There is super weird bug that elif below works with original button's text
                if self.text() == "Diagnostics":
                    classifier = ClassificationResults(files=files, settings=settings, models_info=model_settings,
                                                       diagnostics=True)
                    outputs = classifier.run_diagnostics()
                else:
                    classifier = ClassificationResults(files=files, settings=settings, models_info=model_settings)
                    outputs = classifier.get_outputs()
                    stack.widget(2).set_items(files)
                stack.widget(2).set_inputs(outputs)
                stack.setCurrentIndex(2)
            else:
                QMessageBox.about(self, "Warning", "Files are not selected")

        def set_output_directory(stack: QStackedWidget) -> None:
            """
            Args:
                stack (QStackedWidget): Stack of widgets.
            Returns:
                None
            """
            directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if directory:
                stack.widget(3).children()[0].children()[0].children()[6].setText(directory)    # Meh, ugly.
            settings = stack.widget(3)
            settings.results = directory

        def save_visuals(stack: QStackedWidget) -> None:
            """
            Args:
                stack (QStackedWidget): Stack of widgets.
            Returns:
                None
            """
            stack.currentWidget().save_visuals()

        windows = get_top_level_parent(self)

        if self.text() == "Prediction" or self.text() == "Tool\nDiagnostics":
            self.clicked.connect(lambda: show_file_selection_window(windows))
        elif self.text() == "Settings":
            self.clicked.connect(lambda: windows.setCurrentIndex(3))
        elif self.text() == "Menu":
            self.clicked.connect(lambda: clear(windows))
        elif self.text() == "Clear Data":
            self.clicked.connect(lambda: windows.widget(1).children()[0].clear() if windows.currentIndex() == 1 else
                                 windows.widget(4).children()[1].clear())
        elif self.text() == "Next":
            self.clicked.connect(lambda: show_results(windows))
        elif self.text() == "Apply":
            self.clicked.connect(lambda: windows.widget(3).update_config())
        elif self.text() == "Training":
            self.clicked.connect(lambda: windows.setCurrentIndex(4))
        elif self.text() == "Train":
            self.clicked.connect(lambda: windows.widget(4).begin_training())
        elif self.text() == "...":
            self.clicked.connect(lambda: set_output_directory(windows))
        elif self.text() == "Save Visuals":
            self.clicked.connect(lambda: save_visuals(windows))


class ComboBox(QComboBox, Widget):
    """
    Modified QComboBox class, which inherits set_geometry and set_class from Widget parent class.
    """

    def __init__(self, *args, **kwargs):
        super(QComboBox, self).__init__(widget=Styles.ComboBox, *args, **kwargs)


class CheckableComboBox(QComboBox, Widget):     # From StackOverflow
    # TODO: Add painting with number of chose items in the box
    def __init__(self, *args, **kwargs) -> None:
        super(QComboBox, self).__init__(widget=Styles.ComboBox, *args, **kwargs)
        self.view().pressed.connect(self.__handle_item_pressed)

    def __handle_item_pressed(self, index) -> None:
        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Unchecked)
        else:
            item.setCheckState(Qt.CheckState.Checked)

    def __item_checked(self, index) -> bool:
        item = self.model().item(index, 0)
        return item.checkState() == Qt.CheckState.Checked
    # TODO: Add method to remove checked state for all items

    def set_checked_items(self, items: list) -> None:
        all_items = [self.itemText(i) for i in range(self.count())]
        for item in all_items:
            if item in items:
                self.model().item(all_items.index(item)).setCheckState(Qt.CheckState.Checked)

    def get_check_items(self) -> list:
        checked_items = []
        for i in range(self.count()):
            if self.__item_checked(i):
                checked_items.append(self.model().item(i, 0).text())
        return checked_items


class FileBox(QListWidget, Widget):
    """
    Modified QListWidget class, which inherits set_geometry and set_class from Widget parent class.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(FileBox, self).__init__(widget=Styles.DropArea, *args, **kwargs)


class DropBox(QListWidget, Widget):
    """
    Modified QListWidget class, which inherits set_geometry and set_class from Widget parent class. Additional
    functionality includes possibility to drag and drop files directly into the widget, as well as opening file
    selection dialog window upon click.
    """

    # TODO: Add possibility to remove unwanted files
    def __init__(self, *args, **kwargs) -> None:
        super(DropBox, self).__init__(widget=Styles.DropArea, *args, **kwargs)
        self.setAcceptDrops(True)

    def get_files(self) -> list:
        return [self.item(x).text() for x in range(self.count())]

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()

            links = [str(url.toLocalFile()) if url.isLocalFile() else str(url.toString()) for url in
                     event.mimeData().urls()]
            self.addItems(links)
        else:
            event.ignore()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count() == 0:
            self.setObjectName("drop")
            painter = QPainter(self.viewport())
            painter.save()
            col = self.palette().placeholderText().color()
            painter.setPen(col)
            fm = self.fontMetrics()
            elided_text = fm.elidedText("Drop Files or Click to Select", Qt.TextElideMode.ElideRight,
                                        self.viewport().width())
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, elided_text)
            painter.restore()
        else:
            self.setObjectName("added")
        self.set_style(css=Styles.DropArea.value)

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNames(self, initialFilter="CSV Files (*.csv);; TSV Files (*.tsv);; "
                                                                    "FCS files (*.fcs)")
        self.addItems(files)
