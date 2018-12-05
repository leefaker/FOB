# encoding:utf-8
import sys
import write_result_csv
from PyQt5.QtGui import  *
from PyQt5.QtWidgets import *
import os
import numpy as np
import cv2
import csv
import match

classes = {'N','P','WP'}
class ShowWindow(QWidget):

    def __init__(self):

        super(ShowWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.button_size_height=30
        self.button_size_width=150
        self.re_list={}
        self.img_root=''
        self.img_list=[]
        self.img=QImage()
        self.reimg=QLabel(self)
        self.reimg.setFixedSize(360, 180)
        self.inputLabel = QLabel(self)
        #self.processLabel=QLabel("检测进度")
        #self.processLabel.setFixedSize(70,15)
        self.index=0
        self.openButton = QPushButton("打开图片")
        self.caculateButton=QPushButton("检测")
        self.nextButton=QPushButton('下一张')
        self.preButton=QPushButton('上一张')
        self.viewreButton = QPushButton('查看检测记录')
        self.empty=QLabel()
        self.empty.setFixedSize(10,15)
        self.probar=QProgressBar()
        self.probar.setRange(0,100)
        self.probar.setValue(0)
        self.probar.setFormat("0.0%")
        self.batch_operationButton=QPushButton('批量操作')
        self.setGeometry(100,100,1200,800)#宽，高
        self.image=QLabel(self)
        self.image.setFixedSize(800,700)
        self.result = QLabel("检测结果为：")
        self.result.setFixedSize(150, 15)
        self.la=QPixmap('loading.png')
        self.image.setPixmap(self.la.scaled(self.image.width(), self.image.height()))
        self.batch_operationButton.clicked.connect(self.batch_operation)
        self.openButton.clicked.connect(self.openimage)
        self.caculateButton.clicked.connect(self.caculate)
        self.viewreButton.clicked.connect(self.viewre)
        self.nextButton.clicked.connect(self.next)
        self.preButton.clicked.connect(self.pre)

        # self.batch_operationButton.setFixedSize(self.button_size_width,self.button_size_height)
        # self.openButton.setFixedSize(self.button_size_width,self.button_size_height)
        # self.caculateButton.setFixedSize(self.button_size_width,self.button_size_height)
        # self.viewreButton.setFixedSize(self.button_size_width,self.button_size_height)
        # self.nextButton.setFixedSize(self.button_size_width,self.button_size_height)
        # self.preButton.setFixedSize(self.button_size_width,self.button_size_height)







        batchLayout=QHBoxLayout()
        batchLayout.addWidget(self.batch_operationButton)
        batchLayout.addWidget(self.probar)

        oriLayout = QHBoxLayout()
        oriLayout.addWidget(self.preButton)
        oriLayout.addWidget(self.nextButton)

        singleLayout = QVBoxLayout()
        singleLayout.addWidget(self.openButton,1)
        singleLayout.addWidget(self.empty)
        singleLayout.addLayout(oriLayout)
        singleLayout.addWidget(self.reimg)
        singleLayout.addWidget(self.result)
        # singleLayout.addWidget(self.caculateButton)
        # singleLayout.addWidget(self.viewreButton)
        inspectLayout=QVBoxLayout()
        inspectLayout.addWidget(self.caculateButton)
        inspectLayout.addWidget(self.viewreButton)

        opLayout = QVBoxLayout()
        opLayout.addLayout(batchLayout)
        opLayout.addLayout(singleLayout)
        opLayout.addLayout(inspectLayout)



        imgLayout=QVBoxLayout()
        imgLayout.addWidget(self.inputLabel)
        imgLayout.addWidget(self.image)

        mainLayout=QHBoxLayout()
        mainLayout.addLayout(imgLayout)
        mainLayout.addLayout(opLayout)
        self.setLayout(mainLayout)
        self.setWindowTitle('FOB检测工具')
        self.show()


    def openimage(self):
        self.index=0
        self.re_list.clear()
        self.img_list.clear()
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.BMP;;All Files(*)")
        jpg = QPixmap(imgName)
        if imgName=='':
            #QMessageBox.information(self,"tips","please open an package")
            return
        self.img.load(imgName)#.scaled(self.image.width(), self.image.height())
        self.image.setPixmap(QPixmap.fromImage(self.img.scaled(self.image.width(), self.image.height())))
        self.img_root=os.path.split(imgName)[0]
        self.inputLabel.setText(imgName)
        for name in os.listdir(self.img_root):
            if name.split('.')[-1]=='jpg' or name.split('.')[-1]=='png'or name.split('.')[-1]=='bmp':
                self.img_list.append(name)
        for i in range(0,len(self.img_list)):
            if self.img_list[i]==os.path.split(imgName)[-1]:
                self.index=i
                break

    def caculate(self):
        if (len(self.img_list)==0):
            QMessageBox.information(self, "tip", "there is no picture to examine！")
            return
        image_name=self.img_list[self.index]
        img_path=self.img_root+'/'+self.img_list[self.index]
        img_cv=cv2.imread(img_path)
        img_ori=match.get_ori(img_cv)
        #cv2.imshow('he',img_ori)
        qimg=QImage(img_ori.shape[1],img_ori.shape[0],QImage.Format_RGB888)
        for i in range(img_ori.shape[0])  :
            for j in  range (img_ori.shape[1]):
                #qimg.setColor(0,qRgb(img_ori[i][j][0],img_ori[i][j][1],img_ori[i][j][2]))
                qimg.setPixel(j,i,qRgb(img_ori[i][j][2],img_ori[i][j][1],img_ori[i][j][0]))
        self.reimg.setPixmap(QPixmap.fromImage(qimg))#.scaled(self.reimg.width(), self.reimg.height())))
        predict_label=write_result_csv.predict(img_cv)
        predict_label_string = " "
        if predict_label == 1:
            predict_label_string = "N"
        if predict_label == 2:
            predict_label_string = "P"
        if predict_label == 3:
            predict_label_string = "WP"
        self.result.setText("检测结果为："+predict_label_string)
        if image_name not in self.re_list.keys():
            self.re_list[image_name]=predict_label_string

    def next(self):
        if self.img_list:
            if self.index==len(self.img_list)-1:
                QMessageBox.information(self,"tip","this is the last picture")
            else:
                img_path=self.img_root+'/'+self.img_list[self.index+1]
                self.index+=1
                self.inputLabel.setText(img_path)
                self.img.load(img_path)  # .scaled(self.image.width(), self.image.height())
                self.image.setPixmap(QPixmap.fromImage(self.img.scaled(self.image.width(), self.image.height())))
        else:
            QMessageBox.information(self, "tip", "please open an picture!")
    def pre(self):
        if self.img_list:
            if self.index==0:
                QMessageBox.information(self,"tip","this is the first picture")
            else:
                img_path=self.img_root+'/'+self.img_list[self.index-1]
                self.index-=1
                self.inputLabel.setText(img_path)
                self.img.load(img_path)  # .scaled(self.image.width(), self.image.height())
                self.image.setPixmap(QPixmap.fromImage(self.img).scaled(self.image.width(), self.image.height()))
        else:
            QMessageBox.information(self, "tip", "please open an picture!")

    def batch_operation(self):
        self.re_list.clear()
        dialog=QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dir= dialog.getExistingDirectory(self, "Open Directory","/home",QFileDialog.ShowDirsOnly|
                                                QFileDialog.DontResolveSymlinks)
        # 防止没有选择文件夹
        if dir=="":
            return
        for root, dirs, files_N in os.walk(dir):
            break
        if len(files_N)==0:
            QMessageBox.information(self, "tip", "文件夹内不存在.bmp文件！")
            return
        path_pre = os.path.split(dir)[0]
        f = path_pre + u'/' + 'cut_image'
        stu2 = ["Name", "Value"]

        num_bmp=0
        for image_name in files_N:
            style = image_name.split('.')[-1]
            if style == 'bmp' :
                num_bmp = num_bmp+1

        if num_bmp!= 0:
            csv_path=path_pre + u'/' +os.path.split(dir)[-1]+'_predict.csv'
            if os.path.exists(csv_path):
                os.remove(csv_path)
            csv_file = open(csv_path, 'a', newline='')
            csv_write = csv.writer(csv_file, dialect='excel')
            csv_write.writerow(stu2)
        else:
            QMessageBox.information(self, "tip", "文件夹内不存在.bmp文件！")
            return
        self.probar.setRange(0,num_bmp)

        path_pre = os.path.split(dir)[0]
        f = path_pre + u'/' + 'cut_image'
        # if not os.path.exists(f):
        #      os.makedirs(f)

        num=0
        for image_name in files_N:
            style = image_name.split('.')[-1]
            if style != 'bmp' or style == '':
                continue
            image_path=dir+u'/'+image_name
            image=cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),-1)
            #cv2.imwrite(f+u'/'+image_name,match.get_ori(image))
            predict_label=write_result_csv.predict(image)
            predict_label_string = " "
            if predict_label == 1:
                predict_label_string = "N"
            if predict_label == 2:
                predict_label_string = "P"
            if predict_label == 3:
                predict_label_string = "WP"
            stu1 = [image_name, predict_label_string]
            csv_write.writerow(stu1)
            num=num+1
            self.probar.setValue(num)
            if num==len(files_N):
                self.probar.setValue(0)
                QMessageBox.information(self, "tip", str(num)+"张图片检测完毕！"+'\n'+"检测结果保存在打开的图片文件夹目录下")
                num=0
            self.probar.setFormat("%.1f%%" % float(num / len(files_N) * 100))
            self.re_list[image_name]=predict_label_string
    def viewre(self):
        qdialog=QDialog(self)
        qdialog.setFixedSize(500, 600)
        qwidget=QWidget(qdialog)
        qwidget.setFixedSize(490,600)
        qtext = QTextEdit(qwidget)
        qtext.setFixedSize(490,600)
        re_list = "图片名称:" + '\t' + "检测结果"
        for key in self.re_list:
            re_list = re_list + '\n' + key + '\t' + self.re_list[key]
        qtext.setPlainText(re_list)
        qdialog.setWindowTitle("检测记录")
        mainlayout=QVBoxLayout(qwidget)
        qwidget.setLayout(mainlayout)
        qdialog.exec()








if __name__ =='__main__':
    print(1)
    app = QApplication(sys.argv)
    print("1")
    ex = ShowWindow()
    sys.exit(app.exec_())








