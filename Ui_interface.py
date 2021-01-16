# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\You\Desktop\software\interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(741, 593)
        MainWindow.setMinimumSize(QtCore.QSize(741, 593))
        MainWindow.setMaximumSize(QtCore.QSize(741, 593))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("c:\\Users\\You\\Desktop\\software\\icon/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 721, 521))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.mainFrame = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.mainFrame.setContentsMargins(0, 0, 0, 0)
        self.mainFrame.setObjectName("mainFrame")
        self.fileinfo = QtWidgets.QVBoxLayout()
        self.fileinfo.setObjectName("fileinfo")
        self.fileNameLayout = QtWidgets.QHBoxLayout()
        self.fileNameLayout.setObjectName("fileNameLayout")
        self.fileNameLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.fileNameLabel.setObjectName("fileNameLabel")
        self.fileNameLayout.addWidget(self.fileNameLabel)
        self.fileNameShow = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.fileNameShow.setTextFormat(QtCore.Qt.AutoText)
        self.fileNameShow.setObjectName("fileNameShow")
        self.fileNameLayout.addWidget(self.fileNameShow)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.fileNameLayout.addItem(spacerItem)
        self.fileinfo.addLayout(self.fileNameLayout)
        self.uploadDateLayout = QtWidgets.QHBoxLayout()
        self.uploadDateLayout.setObjectName("uploadDateLayout")
        self.uploadDateLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.uploadDateLabel.setObjectName("uploadDateLabel")
        self.uploadDateLayout.addWidget(self.uploadDateLabel)
        self.dateTimeEdit = QtWidgets.QDateTimeEdit(self.verticalLayoutWidget)
        self.dateTimeEdit.setObjectName("dateTimeEdit")
        self.uploadDateLayout.addWidget(self.dateTimeEdit)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.uploadDateLayout.addItem(spacerItem1)
        self.fileinfo.addLayout(self.uploadDateLayout)
        self.timeCountLayout = QtWidgets.QHBoxLayout()
        self.timeCountLayout.setObjectName("timeCountLayout")
        self.timeCountLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.timeCountLabel.setObjectName("timeCountLabel")
        self.timeCountLayout.addWidget(self.timeCountLabel)
        self.timeCountShow = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.timeCountShow.setObjectName("timeCountShow")
        self.timeCountLayout.addWidget(self.timeCountShow)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.timeCountLayout.addItem(spacerItem2)
        self.fileinfo.addLayout(self.timeCountLayout)
        self.mainFrame.addLayout(self.fileinfo)
        self.segline = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.segline.setFrameShape(QtWidgets.QFrame.HLine)
        self.segline.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.segline.setObjectName("segline")
        self.mainFrame.addWidget(self.segline)
        self.imginfo = QtWidgets.QHBoxLayout()
        self.imginfo.setObjectName("imginfo")
        self.srcimginfo = QtWidgets.QVBoxLayout()
        self.srcimginfo.setObjectName("srcimginfo")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.srcimgshow = QtWidgets.QGraphicsView(self.verticalLayoutWidget)
        self.srcimgshow.setObjectName("srcimgshow")
        self.verticalLayout.addWidget(self.srcimgshow)
        self.uploadButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.uploadButton.setObjectName("uploadButton")
        self.verticalLayout.addWidget(self.uploadButton)
        self.srcimginfo.addLayout(self.verticalLayout)
        self.imginfo.addLayout(self.srcimginfo)
        self.resultinfo = QtWidgets.QVBoxLayout()
        self.resultinfo.setObjectName("resultinfo")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.resultimgshow = QtWidgets.QGraphicsView(self.verticalLayoutWidget)
        self.resultimgshow.setObjectName("resultimgshow")
        self.verticalLayout_2.addWidget(self.resultimgshow)
        self.saveButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.saveButton.setObjectName("saveButton")
        self.verticalLayout_2.addWidget(self.saveButton)
        self.resultinfo.addLayout(self.verticalLayout_2)
        self.imginfo.addLayout(self.resultinfo)
        self.mainFrame.addLayout(self.imginfo)
        self.saveInfoLayout = QtWidgets.QHBoxLayout()
        self.saveInfoLayout.setObjectName("saveInfoLayout")
        self.saveStateLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.saveStateLabel.setObjectName("saveStateLabel")
        self.saveInfoLayout.addWidget(self.saveStateLabel)
        self.savePathLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.savePathLabel.setObjectName("savePathLabel")
        self.saveInfoLayout.addWidget(self.savePathLabel)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.saveInfoLayout.addItem(spacerItem3)
        self.mainFrame.addLayout(self.saveInfoLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 741, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "乳腺超声肿瘤分割系统"))
        self.fileNameLabel.setText(_translate("MainWindow", "文件名："))
        self.fileNameShow.setText(_translate("MainWindow", "File Name"))
        self.uploadDateLabel.setText(_translate("MainWindow", "日  期："))
        self.timeCountLabel.setText(_translate("MainWindow", "耗  时："))
        self.timeCountShow.setText(_translate("MainWindow", "   秒"))
        self.uploadButton.setText(_translate("MainWindow", "选择文件"))
        self.saveButton.setText(_translate("MainWindow", "保存结果"))
        self.saveStateLabel.setText(_translate("MainWindow", "保存状态："))
        self.savePathLabel.setText(_translate("MainWindow", "结果未保存"))
