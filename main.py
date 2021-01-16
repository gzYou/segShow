import torch
from PIL import Image as PILImage
from monai.transforms import (
    AddChannel,
    Compose,
    ScaleIntensity,
    Resize,
    ToTensor,
)
from monai.data.image_reader import PILReader

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import sys
from PyQt5 import Qt
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QPushButton,QDialog,QGraphicsScene, QApplication, QMainWindow,QFileDialog,QGraphicsPixmapItem
from PyQt5.QtCore import QDate,QDateTime
import Ui_interface
from model import networks



transforms = Compose([
        ScaleIntensity(), 
        Resize(spatial_size=(448,448)),
        # ToTensor()
])

def restore_model(modelS, path):
    modelS.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return modelS

def init_model():
    modelS = networks.UNetComposedLossSupervised()
    return modelS

def get_model():
    model = init_model()
    model = restore_model(model,"./checkpoint/best_55.pth")
    model.eval()
    return model

def pad(image):
    w,h = image.size
    background = PILImage.new('RGB', size=(max(w, h), max(w, h)), color=(127, 127, 127))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image,box)
    return background

def get_image(path):
    image = PILImage.open(path)
    image = pad(image)
    img_array, meta_data = PILReader().get_data(image)
    img_array = img_array.astype(dtype=np.float32)
    img_array = img_array.transpose((2, 0, 1))
    return transforms(img_array)#img_array,transforms(img_array)

def predict(model,image):
    image = AddChannel()(image)
    image = ToTensor()(image)
    
    output,_,_,_,_ = model(image)
    output = (output > 0.5)
    output = output.numpy()

    return output

def chooseFile():
    path,_ = QFileDialog.getOpenFileName()
    print(path)
    return path

def cvshow(img,name):
    cv2.namedWindow(name) 
    cv2.imshow(name, img) 
    cv2.waitKey (0) 
    cv2.destroyAllWindows()

class MyUI(Ui_interface.Ui_MainWindow):
    def __init__(self,MainWindow):
        self.mainwindow = MainWindow
        self.setupUi(self.mainwindow)
        self.image_path = ""
        self.image = None
        self.result = None
        self.mask = None
        self.model = get_model()
        self.uploadButton.clicked.connect(self.show)
        self.saveButton.clicked.connect(self.save)

    def chooseFile(self):
        self.image_path, _ = QFileDialog.getOpenFileName()

    def show_upinfo(self):
        self.fileNameShow.setText(self.image_path)
        self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())
        self.dateTimeEdit.setDate(QDate.currentDate())

    def _show(self,src):

        srcFrame = QImage((src*255).astype(np.int8),src.shape[1],src.shape[0],src.shape[1]*3,QImage.Format_RGB888)
        srcPix = QPixmap.fromImage(srcFrame).scaled(352,352)
        srcItem = QGraphicsPixmapItem(srcPix)
        srcScene = QGraphicsScene() 
        srcScene.addItem(srcItem)

        return srcScene


    def show_src(self):
        self.image = get_image(self.image_path)
        src = self.image.transpose(1,2,0).copy()
        # cvshow(src,"src")
        srcScene = self._show(src)
        self.srcimgshow.setScene(srcScene)

    def show_result(self,):
        contours, _ = cv2.findContours(self.result[0,0,:,:].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        image = self.image.transpose(1,2,0).copy()
        for i in range(0,len(contours)):
            cv2.polylines(image,contours[i],True,(1,0,0),2)
        self.mask = image
        resScene = self._show(image)
        # cvshow(image,"res")
        self.resultimgshow.setScene(resScene)

    def failedSaveDialog(self):
        title = "保存失败"
        content = "请先选择文件"
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, title,content)
        box.exec_()

    def save(self):
        if self.mask is None:
            self.failedSaveDialog()
        else:
            savepath, ok = QFileDialog.getSaveFileName( caption="文件保存", directory="/", filter="图片文件 (*.png);;(*.jpeg)")
            saveimage = cv2.cvtColor(self.mask*255,cv2.COLOR_RGB2BGR)
            cv2.imwrite(savepath,saveimage)
            self.savePathLabel.setText("已保存至"+savepath)

        

    def show(self):
        self.chooseFile()
        if self.image_path != "":
            start = time.time()
            self.show_upinfo()
            self.show_src()
            self.result = predict(self.model,self.image)
            self.show_result()
            end = time.time()
            timeconsuming = end - start
            self.timeCountShow.setText(str(timeconsuming)+"秒")
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    myui = MyUI(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())