from utilities.classification_utils import ClassificationTraining, ClassificationResults, AutoEncoder
from utilities.data_preparation import FilePreparation

from gui.widgets import Widget, DropBox, Button, EditLine, CheckBox, MessageBox


class FileSelector(Widget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = "Choose Files"
        self.action = False

    def __init_ui(self):
        self.setWindowTitle(self.title)
        DropBox(obj_name="drop", geometry=[46, 43, 808, 450], parent=self)
        Button(text="Menu", obj_name="standard", geometry=[46, 518, 200, 60], parent=self)
        Button(text="Clear Data", obj_name="standard", geometry=[349, 518, 200, 60], parent=self)
        Button(text="Predict", obj_name="standard", geometry=[652, 518, 200, 60], parent=self)
        if self.action == "Tool\nDiagnostics":
            self.children()[3].setText("Diagnose")
        elif self.action == "Training":
            model_name = EditLine(obj_name="input", geometry=[46, 43, 638, 30], parent=self)
            model_name.setPlaceholderText("Enter model name")
            CheckBox(text="Autoencoder", obj_name="use_ae", geometry=[703, 43, 150, 30], parent=self)
            self.children()[0].setGeometry(46, 93, 808, 400)
            self.children()[3].setText("Train")

    def set_action(self, action: str) -> None:
        self.action = action
        [children.setParent(None) for children in self.children()]
        self.__init_ui()

    def run_action(self) -> None:
        if self.children()[0].count() != 0:
            settings = self.stack.widget(0).get_settings()
            models_info = self.stack.widget(0).get_models_info()
            if self.action == "Training":
                model_name = self.children()[4].text()
                if model_name == "":
                    MessageBox.about(self, "Warning", "Model name is not provided")
                    return
                files = self.children()[0].get_files()
                if self.children()[5].isChecked():
                    # TODO: Next lines are repeated in the interface.py file. Refactor
                    file_prep = FilePreparation(files=files, settings=settings, models_info=models_info, training=True)
                    dataframe = file_prep.get_aggregated()
                    labels = file_prep.get_labels()
                    features = file_prep.get_features()
                    autoencoder = AutoEncoder(settings=settings, models_info=models_info)
                    autoencoder.retrain(dataframe=dataframe, labels=labels, name=model_name, columns=features)
                else:
                    training = ClassificationTraining(files=files, model_name=model_name, settings=settings,
                                                      models_info=models_info)
                    training.run_training()
            else:
                diagnostics = False if self.action == "Prediction" else True
                files = self.children()[0].get_files()
                models_info.classifier_name = settings.model
                models_info.classifier_name = settings.model
                models_info.autoencoder_name = settings.autoencoder
                classifier = ClassificationResults(files=files, settings=settings, models_info=models_info,
                                                   diagnostics=diagnostics)
                outputs = classifier.run_diagnostics()
                if not diagnostics:
                    self.stack.widget(2).set_items(files)
                self.stack.widget(2).set_inputs(outputs)
                self.stack.setCurrentIndex(2)
        else:
            MessageBox.about(self, "Warning", "Files are not selected")
