import os, sys
from PyQt5 import QtCore, QtGui, QtWidgets
from tkinter import *
import samples.balloon2.balloon2_splash as splash
import samples.balloon2.balloon2_test as test



class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        # ********** ********** Background ********** **********
        self.setObjectName("Test Program")
        self.resize(1210, 850)
        self.setFixedSize(1210,850)
        self.setStyleSheet("background-color: rgb(225, 225, 225);")

        # ********** ********** button ********** **********
        self.btn1_folder = QtWidgets.QPushButton(self)
        self.btn1_folder.setGeometry(QtCore.QRect(170, 10, 51, 31))
        self.btn1_folder.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                         "image: url(folder.jpg);")
        self.btn1_folder.setText("")
        self.btn1_folder.setObjectName("btn1_folder")
        self.btn1_folder.clicked.connect(self.openImagePath)

        self.btn2_folder = QtWidgets.QPushButton(self)
        self.btn2_folder.setGeometry(QtCore.QRect(130, 430, 51, 31))
        self.btn2_folder.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                         "image: url(folder.jpg);")
        self.btn2_folder.setText("")
        self.btn2_folder.setObjectName("btn2_folder")
        self.btn2_folder.clicked.connect(self.openModelPath)

        self.btn3_start = QtWidgets.QPushButton(self)
        self.btn3_start.setGeometry(QtCore.QRect(10, 520, 290, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn3_start.setFont(font)
        self.btn3_start.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                        "color: rgb(0, 0, 0);")
        self.btn3_start.setObjectName("btn3_start")
        self.btn3_start.clicked.connect(self.splashStart)

        self.btn4_start = QtWidgets.QPushButton(self)
        self.btn4_start.setGeometry(QtCore.QRect(311, 520, 290, 41))
        self.btn4_start.setFont(font)
        self.btn4_start.setStyleSheet("background-color: rgb(200, 200, 200);\n"
                                        "color: rgb(0, 0, 0);")
        self.btn4_start.setObjectName("btn4_start")
        self.btn4_start.clicked.connect(self.testStart)

        # ********** ********** text ********** **********
        self.text1_image = QtWidgets.QTextBrowser(self)
        self.text1_image.setGeometry(QtCore.QRect(10, 50, 591, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        self.text1_image.setFont(font)
        self.text1_image.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.text1_image.setFrameShape(QtWidgets.QFrame.Box)
        self.text1_image.setFrameShadow(QtWidgets.QFrame.Plain)
        self.text1_image.setObjectName("text1_image")

        self.text2_model = QtWidgets.QTextBrowser(self)
        self.text2_model.setGeometry(QtCore.QRect(10, 470, 591, 41))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        self.text2_model.setFont(font)
        self.text2_model.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.text2_model.setFrameShape(QtWidgets.QFrame.Box)
        self.text2_model.setFrameShadow(QtWidgets.QFrame.Plain)
        self.text2_model.setObjectName("text2_model")

        self.text3_train = QtWidgets.QTextBrowser(self)
        self.text3_train.setGeometry(QtCore.QRect(10, 570, 591, 271))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        font.setPointSize(10)
        self.text3_train.setFont(font)
        self.text3_train.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.text3_train.setFrameShape(QtWidgets.QFrame.Box)
        self.text3_train.setFrameShadow(QtWidgets.QFrame.Plain)
        self.text3_train.setObjectName("text3_train")

        # ********** ********** tree ********** **********
        self.tree1_image = QtWidgets.QTreeView(self)
        self.tree1_image.setGeometry(QtCore.QRect(10, 100, 591, 321))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift Light")
        self.tree1_image.setFont(font)
        self.tree1_image.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.tree1_image.setFrameShape(QtWidgets.QFrame.Box)
        self.tree1_image.setFrameShadow(QtWidgets.QFrame.Plain)
        self.tree1_image.setObjectName("tree1_image")

        # ********** ********** label ********** **********
        self.label1_text = QtWidgets.QLabel(self)
        self.label1_text.setGeometry(QtCore.QRect(10, 10, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label1_text.setFont(font)
        self.label1_text.setObjectName("label1_text")

        self.label2_text = QtWidgets.QLabel(self)
        self.label2_text.setGeometry(QtCore.QRect(10, 430, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label2_text.setFont(font)
        self.label2_text.setObjectName("label2_text")

        self.label3_image = QtWidgets.QLabel(self)
        self.label3_image.setGeometry(QtCore.QRect(610, 10, 591, 411))
        self.label3_image.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label3_image.setFrameShape(QtWidgets.QFrame.Box)
        self.label3_image.setLineWidth(1)
        self.label3_image.setText("")
        self.label3_image.setObjectName("label3_image")

        self.label4_image = QtWidgets.QLabel(self)
        self.label4_image.setGeometry(QtCore.QRect(610, 430, 591, 411))
        self.label4_image.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label4_image.setFrameShape(QtWidgets.QFrame.Box)
        self.label4_image.setLineWidth(1)
        self.label4_image.setText("")
        self.label4_image.setObjectName("label4_image")


        self.test_image = "."
        self.test_model = "."



        # ********** ********** Execute ********** **********
        self.center()  # ** 창을 화면의 정 가운데에 위치 **
        self.retranslateUi(self)
        self.show()


    def center(self):
        ct = self.frameGeometry()
        ct2 = QtWidgets.QDesktopWidget().availableGeometry().center()
        ct.moveCenter(ct2)
        self.move(ct.topLeft())


    def retranslateUi(self, TT):
        _translate = QtCore.QCoreApplication.translate
        TT.setWindowTitle(_translate("TT", "Test Program"))
        self.btn3_start.setText(_translate("TT", "Splash"))
        self.btn4_start.setText(_translate("TT", "Object Recognition"))
        self.label1_text.setText(_translate("TT", " Image File Path"))
        self.label2_text.setToolTip(_translate("TT", "<html><head/><body><p><br/></p></body></html>"))
        self.label2_text.setText(_translate("TT", " Model Path"))


    def openImagePath(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '')
        image = QtGui.QImage(filename)
        if image.width() > 591:
            image = image.scaledToWidth(591)
        if image.height() > 411:
            image = image.scaledToHeight(411)
        self.label3_image.setPixmap(QtGui.QPixmap.fromImage(image))
        self.label3_image.setAlignment(QtCore.Qt.AlignCenter)
        self.test_image = filename
        self.text1_image.setText(self.test_image)

        path = filename + "/../"
        self.dirModel2 = QtWidgets.QFileSystemModel()
        self.dirModel2.setRootPath(path)
        self.tree1_image.setModel(self.dirModel2)
        self.tree1_image.setRootIndex(self.dirModel2.index(path))
        self.tree1_image.clicked.connect(self.openImage)


    def openImage(self, index):
        filename = self.dirModel2.fileInfo(index).absoluteFilePath()
        image = QtGui.QImage(filename)
        if image.width() > 591:
            image = image.scaledToWidth(591)
        if image.height() > 411:
            image = image.scaledToHeight(411)

        self.label3_image.setPixmap(QtGui.QPixmap.fromImage(image))
        self.label3_image.setAlignment(QtCore.Qt.AlignCenter)
        self.test_image = filename
        self.text1_image.setText(self.test_image)


    def openModelPath(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '')
        self.text2_model.setText(filename)
        self.test_model = filename


    def splashStart(self):
        self.new_filename, text_image, text_molded_images, text_image_metas, text_anchors\
            = splash.training(self.test_image, self.test_model)
        self.text3_train.setText(text_image)
        self.text3_train.append(text_molded_images)
        self.text3_train.append(text_image_metas)
        self.text3_train.append(text_anchors)

        image = QtGui.QImage(self.new_filename)
        image = image.scaledToHeight(411)
        if image.width() > 591:
            image = image.scaledToWidth(591)

        self.label4_image.setPixmap(QtGui.QPixmap.fromImage(image))
        self.label4_image.setAlignment(QtCore.Qt.AlignCenter)


    def testStart(self):
        self.new_filename, text_image, text_molded_images, text_image_metas, text_anchors\
            = test.training(self.test_image, self.test_model)
        self.text3_train.setText(text_image)
        self.text3_train.append(text_molded_images)
        self.text3_train.append(text_image_metas)
        self.text3_train.append(text_anchors)

        image = QtGui.QImage(self.new_filename)
        image = image.scaledToHeight(600)
        # image = image.scaledToHeight(411)
        # if image.width() > 591:
        #     image = image.scaledToWidth(591)

        self.label4_image.setPixmap(QtGui.QPixmap.fromImage(image))
        self.label4_image.setAlignment(QtCore.Qt.AlignCenter)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())