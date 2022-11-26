from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QListWidget, QFileDialog, QStackedWidget, QMessageBox, \
    QFrame, QComboBox, QLineEdit, QTextEdit, QCheckBox, QInputDialog
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QPainter


class Widget(QWidget):
    """
    Base class for all the widgets. Inherits everything from QWidget and has additional attributes and methods. Allows
    to set the object name, geometry and name at the initialization.

    Attributes:
    ----------
    name: str
        Name of the widget used to differentiate widgets to fill in the settings attributes.
    stack: QStackedWidget
        Stack of widgets used to switch between them.
    """

    name = None
    stack = None

    def __init__(self, obj_name: str = None, geometry: list = None, name: str = None, stack=None, *args, **kwargs):
        """
        Args:
            obj_name (str, optional): Object name of the widget. Defaults to None.
            geometry (list, optional): Geometry of the widget. Defaults to None.
            name (str, optional): Name of the widget. Defaults to None.
            stack (QStackedWidget, optional): Stack of widgets. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.center()
        if obj_name:
            self.setObjectName(obj_name)
        if geometry:
            self.setGeometry(*geometry)
        if name:
            setattr(self, name, name)
            self.name = name
        if stack is not None:
            self.stack = stack
            self.stack.addWidget(self)
        with open("gui/widgets.css", "r") as file:
            style = file.read()
            self.setStyleSheet(style)

    def center(self) -> None:
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class Stack(QStackedWidget, Widget):
    """
    Stack of widgets. Inherits from QStackedWidget and Widget.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(Stack, self).__init__(*args, **kwargs)


class Label(QLabel, Widget):
    """
    Label widget. Inherits from QLabel and Widget. Additionally, depending on the object name, it sets the alignment.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(Label, self).__init__(*args, **kwargs)
        if self.objectName() == "tittle" or self.objectName() == "version" or self.objectName() == "overlay":
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class MessageBox(QMessageBox, Widget):
    """
    Message box widget. Inherits from QMessageBox and Widget.
    """

    def __init__(self, *args, **kwargs):
        super(MessageBox, self).__init__(*args, **kwargs)


class HLine(QFrame, Widget):
    """
    Horizontal line widget. Inherits from QFrame and Widget.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(HLine, self).__init__(*args, **kwargs)
        self.setFrameShape(QFrame.Shape.HLine)


class TextEdit(QTextEdit, Widget):
    """
    Text edit widget. Inherits from QTextEdit and Widget.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TextEdit, self).__init__(*args, **kwargs)


class EditLine(QLineEdit, Widget):
    """
    Edit line widget. Inherits from QLineEdit and Widget.
    """

    def __init__(self, *args, **kwargs):
        super(EditLine, self).__init__(*args, **kwargs)


class Button(QPushButton, Widget):
    """
    Button widget. Inherits from QPushButton and Widget. Additionally, it adds an action to the button depending on the
    text of the button.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(Button, self).__init__(*args, **kwargs)
        self.__add_action()
        if self.text() == "?":
            self.installEventFilter(self)

    def __add_action(self) -> None:
        """
        Adds an action to the button depending on the text of the button.
        Returns:
            None.
        """

        def show_file_selection_window(stack: QStackedWidget) -> None:
            """
            Shows the file selection window. After each "Menu", resets the action to be performed in the window.
            Args:
                stack (QStackedWidget): Stack of widgets.
            Returns:
                None.
            """
            stack.widget(1).set_action(self.text())
            stack.setCurrentIndex(1)

        def clear(stack: QStackedWidget) -> None:
            """
            Clears content of some widgets. In case of result window for training model, kills running tfboard.
            Args:
                stack (QStackedWidget): Stack of widgets.
            Returns:
                None.
            """
            for widget in stack.widget(1).children():
                if isinstance(widget, EditLine) or isinstance(widget, DropBox):
                    widget.clear()
            if self.text() == "Menu":
                if stack.currentIndex() == 4:
                    stack.widget(4).set_values_from_config()
                    stack.setGeometry(*[300, 300, 895, 600])
                elif stack.currentIndex() == 3:
                    stack.widget(3).tf_board.kill()
                elif stack.currentIndex() == 2:
                    stack.widget(2).clear()
                stack.setCurrentIndex(0)
                stack.center()

        def set_output_directory(stack: QStackedWidget) -> None:
            """
            Sets the output directory by raising a file selection window.
            Args:
                stack (QStackedWidget): Stack of widgets.
            Returns:
                None.
            """
            directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if directory:
                for widget in stack.widget(4).children():
                    if widget.name == "results":
                        widget.setText(directory)

        def show_settings(stack: QStackedWidget) -> None:
            """
            Shows the settings window.
            Args:
                stack (QStackedWidget): Stack of widgets.
            Returns:
                None.
            """
            stack.setCurrentIndex(4)
            stack.currentWidget().center()
            stack.setGeometry(*[300, 300, 895, 700])
            stack.center()

        windows = self.parent().stack
        if self.text() == "Prediction" or self.text() == "Tool\nDiagnostics" or self.text() == "Training":
            self.clicked.connect(lambda: show_file_selection_window(windows))
        elif self.text() == "Settings":
            self.clicked.connect(lambda: show_settings(windows))
        elif self.text() == "Help":
            self.clicked.connect(lambda: windows.setCurrentIndex(5))
        elif self.text() == "Menu" or self.text() == "Clear Data":
            self.clicked.connect(lambda: clear(windows))
        elif self.text() == "Predict" or self.text() == "Diagnose" or self.text() == "Train":
            self.clicked.connect(lambda: windows.widget(1).run_action())
        elif self.text() == "Apply":
            self.clicked.connect(lambda: windows.widget(4).update_config())
        elif self.text() == "Change":
            self.clicked.connect(lambda: set_output_directory(windows))
        elif self.text() == "MSE" or self.text() == "Predictions":
            self.clicked.connect(lambda: windows.currentWidget().change_plot(plot_type=self.text()))
        elif self.text() == "Adjust MSE":
            self.clicked.connect(lambda: windows.currentWidget().adjust_mse())
        elif self.text() == "Save Results":
            self.clicked.connect(lambda: windows.currentWidget().save_results())

    def eventFilter(self, object, event) -> bool:
        """
        Event filter for the help button. It shows a message box with the model metadata.
        """
        classifier = self.parent().children()[-2]
        autoencoder = self.parent().children()[-1]
        if event.type() == QEvent.Type.HoverEnter:
            autoencoder.show() if self.name == "ae" else classifier.show()
        elif event.type() == QEvent.Type.HoverLeave:
            autoencoder.hide() if self.name == "ae" else classifier.hide()
        return False


class InputDialog(QInputDialog, Widget):
    """
    Input dialog widget. Inherits from QInputDialog and Widget.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(InputDialog, self).__init__(*args, **kwargs)


class CheckBox(QCheckBox, Widget):
    """
    Check box widget. Inherits from QCheckBox and Widget.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(CheckBox, self).__init__(*args, **kwargs)


class ComboBox(QComboBox, Widget):
    """
    Combo box widget. Inherits from QComboBox and Widget.
    """

    def __init__(self, *args, **kwargs):
        super(QComboBox, self).__init__(*args, **kwargs)


class CheckableComboBox(QComboBox, Widget):
    """
    Checkable combo box widget. Inherits from QComboBox and Widget. Allows to select multiple items.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(QComboBox, self).__init__(*args, **kwargs)
        self.view().pressed.connect(self.__handle_item_pressed)

    def __handle_item_pressed(self, index: int) -> None:
        """
        Handles the pressed event of the combo box.
        Args:
            index (int): Index of the item.
        Returns:
            None.
        """
        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Unchecked)
        else:
            if self.name == "vis_channels":
                checked_items = self.get_check_items()
                if len(checked_items) >= int(self.parent().children()[15].currentText()):
                    checked_items.pop(0)
                    self.set_checked_items(checked_items)
            item.setCheckState(Qt.CheckState.Checked)

    def __item_checked(self, index: int) -> bool:
        """
        Returns whether the item at the given index is checked.
        Args:
            index (int): Index of the item.
        Returns:
            bool: True if the item is checked, False otherwise.
        """
        item = self.model().item(index, 0)
        return item.checkState() == Qt.CheckState.Checked

    def item_unchecked(self) -> None:
        """
        In case current number of checked item is more than vis_dims, unchecks first item.
        Returns:
            None
        """
        checked_items = self.get_check_items()
        try:
            if len(checked_items) >= int(self.parent().children()[15].currentText()):
                checked_items.pop(0)
                self.set_checked_items(checked_items)
        except ValueError:
            pass

    def set_checked_items(self, items: list) -> None:
        """
        Sets the checked items in the combo box.
        Args:
            items (list): List of items to be checked.
        Returns:
            None.
        """
        all_items = [self.itemText(i) for i in range(self.count())]
        for item in all_items:
            self.model().item(all_items.index(item)).setCheckState(Qt.CheckState.Unchecked)
            if item in items:
                self.model().item(all_items.index(item)).setCheckState(Qt.CheckState.Checked)

    def get_check_items(self) -> list:
        """
        Returns a list of checked items.
        Returns:
            list: List of checked items.
        """
        checked_items = []
        for i in range(self.count()):
            if self.__item_checked(i):
                checked_items.append(self.model().item(i, 0).text())
        return checked_items


class FileBox(QListWidget, Widget):
    """
    File box widget. Inherits from QListWidget and Widget.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(FileBox, self).__init__(*args, **kwargs)


class DropBox(QListWidget, Widget):
    """
    Drop box widget. Inherits from QListWidget and Widget. Allows to drag and drop files.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(DropBox, self).__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def get_files(self) -> list:
        """
        Returns a list of files in the drop box.
        Returns:
            list: List of files in the drop box.
        """
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

    def paintEvent(self, event) -> None:
        """
        Add a placeholder text if the drop box is empty and changes its object name to adjust stylesheets used.
        """
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
        self.setStyleSheet("QListWidget#drop {font-size: 32px;}")

    def mousePressEvent(self, event) -> None:
        """
        Handles the mouse press event.
        """
        files, _ = QFileDialog.getOpenFileNames(self, initialFilter="CSV Files (*.csv);; TSV Files (*.tsv);; "
                                                                    "FCS files (*.fcs)")
        self.addItems(files)
