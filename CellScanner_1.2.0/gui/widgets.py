from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QListWidget, QFileDialog, QStackedWidget, QMessageBox, \
    QFrame, QComboBox, QLineEdit, QTextEdit, QCheckBox, QInputDialog
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QPainter


class Widget(QWidget):
    name = None
    stack = None

    def __init__(self, obj_name: str = None, geometry: list = None, name: str = None, stack=None, *args, **kwargs):
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

    def __init__(self, *args, **kwargs) -> None:
        super(Stack, self).__init__(*args, **kwargs)


class Label(QLabel, Widget):

    def __init__(self, *args, **kwargs) -> None:
        super(Label, self).__init__(*args, **kwargs)
        if self.objectName() == "tittle" or self.objectName() == "version" or self.objectName() == "overlay":
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class MessageBox(QMessageBox, Widget):

    def __init__(self, *args, **kwargs):
        super(MessageBox, self).__init__(*args, **kwargs)


class HLine(QFrame, Widget):

    def __init__(self, *args, **kwargs) -> None:
        super(HLine, self).__init__(*args, **kwargs)
        self.setFrameShape(QFrame.Shape.HLine)


class TextEdit(QTextEdit, Widget):

    def __init__(self, *args, **kwargs) -> None:
        super(TextEdit, self).__init__(*args, **kwargs)


class EditLine(QLineEdit, Widget):

    def __init__(self, *args, **kwargs):
        super(EditLine, self).__init__(*args, **kwargs)


class Button(QPushButton, Widget):

    def __init__(self, *args, **kwargs) -> None:
        super(Button, self).__init__(*args, **kwargs)
        self.__add_action()
        if self.text() == "?":
            self.installEventFilter(self)

    def __add_action(self) -> None:

        def show_file_selection_window(stack: QStackedWidget) -> None:
            stack.widget(1).set_action(self.text())
            stack.setCurrentIndex(1)

        def clear(stack: QStackedWidget) -> None:
            for widget in stack.widget(1).children():
                if isinstance(widget, EditLine) or isinstance(widget, DropBox):
                    widget.clear()
            if self.text() == "Menu":
                if stack.currentIndex() == 3:
                    stack.widget(3).set_values_from_config()
                    stack.setGeometry(*[300, 300, 895, 600])
                elif stack.currentIndex() == 5:
                    stack.widget(5).tf_board.kill()
                elif stack.currentIndex() == 2:
                    stack.widget(2).clear()
                stack.setCurrentIndex(0)
                stack.center()

        def set_output_directory(stack: QStackedWidget) -> None:
            directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if directory:
                for widget in stack.widget(3).children():
                    if widget.name == "results":
                        widget.setText(directory)

        def show_settings(stack: QStackedWidget) -> None:
            stack.setCurrentIndex(3)
            stack.currentWidget().center()
            stack.setGeometry(*[300, 300, 895, 700])
            stack.center()

        windows = self.parent().stack
        if self.text() == "Prediction" or self.text() == "Tool\nDiagnostics" or self.text() == "Training":
            self.clicked.connect(lambda: show_file_selection_window(windows))
        elif self.text() == "Settings":
            self.clicked.connect(lambda: show_settings(windows))
        elif self.text() == "Help":
            self.clicked.connect(lambda: windows.setCurrentIndex(4))
        elif self.text() == "Menu" or self.text() == "Clear Data":
            self.clicked.connect(lambda: clear(windows))
        elif self.text() == "Predict" or self.text() == "Diagnose" or self.text() == "Train":
            self.clicked.connect(lambda: windows.widget(1).run_action())
        elif self.text() == "Apply":
            self.clicked.connect(lambda: windows.widget(3).update_config())
        elif self.text() == "Change":
            self.clicked.connect(lambda: set_output_directory(windows))
        elif self.text() == "MSE" or self.text() == "Predictions":
            self.clicked.connect(lambda: windows.currentWidget().change_plot(plot_type=self.text()))
        elif self.text() == "Adjust MSE":
            self.clicked.connect(lambda: windows.currentWidget().adjust_mse())
        elif self.text() == "Save Results":
            self.clicked.connect(lambda: windows.currentWidget().save_results())

    def eventFilter(self, object, event) -> bool:
        classifier = self.parent().children()[-2]
        autoencoder = self.parent().children()[-1]
        if event.type() == QEvent.Type.HoverEnter:
            autoencoder.show() if self.name == "ae" else classifier.show()
        elif event.type() == QEvent.Type.HoverLeave:
            autoencoder.hide() if self.name == "ae" else classifier.hide()
        return False


class InputDialog(QInputDialog, Widget):

    def __init__(self, *args, **kwargs) -> None:
        super(InputDialog, self).__init__(*args, **kwargs)


class CheckBox(QCheckBox, Widget):

    def __init__(self, *args, **kwargs) -> None:
        super(CheckBox, self).__init__(*args, **kwargs)


class ComboBox(QComboBox, Widget):

    def __init__(self, *args, **kwargs):
        super(QComboBox, self).__init__(*args, **kwargs)


class CheckableComboBox(QComboBox, Widget):
    # TODO: Add painting with number of chose items in the box
    def __init__(self, *args, **kwargs) -> None:
        super(QComboBox, self).__init__(*args, **kwargs)
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

    def __init__(self, *args, **kwargs) -> None:
        super(FileBox, self).__init__(*args, **kwargs)


class DropBox(QListWidget, Widget):
    # TODO: Add possibility to remove unwanted files
    def __init__(self, *args, **kwargs) -> None:
        super(DropBox, self).__init__(*args, **kwargs)
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
        self.setStyleSheet("QListWidget#drop {font-size: 32px;}")

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNames(self, initialFilter="CSV Files (*.csv);; TSV Files (*.tsv);; "
                                                                    "FCS files (*.fcs)")
        self.addItems(files)
