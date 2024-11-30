import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
from tensorflow.keras.models import load_model

# Define classes
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.WindowModal)
        Dialog.resize(985, 709)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(8)
        sizePolicy.setVerticalStretch(0)
        Dialog.setSizePolicy(sizePolicy)
        
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(280, 20, 421, 211))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(280, 170, 431, 321))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(160, 570, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(160, 650, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(570, 650, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(540, 560, 181, 51))
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        # Connect buttons to functions
        self.pushButton.clicked.connect(self.loadImage)
        self.pushButton_2.clicked.connect(self.classifyFunction)
        self.pushButton_3.clicked.connect(self.trainingFunction)

        # Load the trained model
        try:
            self.model = load_model("my_model_new.h5")
        except Exception as e:
            print(f"Error loading model: {e}")
            QtWidgets.QMessageBox.critical(Dialog, "Error", f"Failed to load model: {e}")

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Brain Tumor Classification"))
        self.label.setText(_translate("Dialog", "BRAIN TUMOUR CLASSIFICATION"))
        self.pushButton.setText(_translate("Dialog", "Browse image"))
        self.pushButton_2.setText(_translate("Dialog", "Classify"))
        self.pushButton_3.setText(_translate("Dialog", "Training"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if fileName:
            print(fileName)
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
            self.label_2.setPixmap(pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def classifyFunction(self):
        if not hasattr(self, 'file') or not self.file:
            QtWidgets.QMessageBox.warning(None, "Error", "Please select an image first.")
            return

        path2 = self.file
        test_image = Image.open(path2)
        test_image = test_image.resize((30, 30))  # Resize image to 30x30 (same as model input)
        test_image = test_image.convert('L')  # Convert to grayscale (L mode)
        test_image = np.array(test_image)
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
        test_image = np.expand_dims(test_image, axis=-1)  # Add channel dimension (single channel)

        # Normalize if needed (depending on how the model was trained)
        test_image = test_image / 255.0  # Example normalization step

        # Predict with the loaded model
        result = self.model.predict(test_image)
        predicted_class_index = np.argmax(result)
        predicted_class = class_names[predicted_class_index]
        print(predicted_class)
        self.textBrowser.setText(predicted_class)

    def trainingFunction(self):
        QtWidgets.QMessageBox.information(None, "Training", "Training functionality is not yet implemented.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
