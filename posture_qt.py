# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'posture_qt.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from qfluentwidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.zoom = 1.5
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(int(1050*self.zoom), int(723*self.zoom))
        MainWindow.resize(1675, 1094)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet('''QWidget{background-color:#99ffff;}''')
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 750, 1050))
        self.label.setObjectName("label")
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setKerning(True)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setLineWidth(0)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("QLabel { background-color:#7FFFD4; }")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1290, 50, 351, 301))
        self.label_2.setObjectName("label_2")
        font = QtGui.QFont()
        font.setFamily('Arial')  # 设置字体为 Arial
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setKerning(True)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setLineWidth(0)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setStyleSheet("QLabel { background-color:#7FFFD4; }")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        # self.widget.setGeometry(QtCore.QRect(int(550*self.zoom+60), int(330*self.zoom-60), int(200*self.zoom), int(200*self.zoom)))
        self.widget.setGeometry(QtCore.QRect(885, 435, 300, 300))
        self.widget.setObjectName("widget")

        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")


        self.widget_plt = QtWidgets.QWidget(self.centralwidget)
        # self.widget_plt.setGeometry(QtCore.QRect(int(550*self.zoom), int(10*self.zoom), int(411*self.zoom-200), int(300*self.zoom)))
        self.widget_plt.setGeometry(QtCore.QRect(825, 15, 416, 450))
        self.widget_plt.setObjectName("widget_plt")
        self.widget_plt.setStyleSheet('''QWidget{background-color:#98F5FF;}''')  # 初音绿
        self.gridLayout_plt = QtWidgets.QGridLayout(self.widget_plt)
        self.gridLayout_plt.setObjectName("gridLayout_plt")

        self.widget_motion = QtWidgets.QWidget(self.centralwidget)
        self.widget_motion.setStyleSheet('''QWidget{background-color:#98F5FF;}''')
        self.widget_motion.setGeometry(QtCore.QRect(825, 715, 416, 350))
        self.widget_motion.setObjectName("widget_motion")
        self.gridLayout_motion = QtWidgets.QGridLayout(self.widget_motion)
        self.gridLayout_motion.setObjectName("gridLayout_motion")

        self.pushButton_1 = PrimaryPushButton(self.widget)
        self.pushButton_1.setObjectName("pushButton_1")
        # self.pushButton_1.setFixedSize(100, 50)
        self.gridLayout.addWidget(self.pushButton_1, 0, 0, 1, 1)

        self.pushButton_2 = PrimaryPushButton(self.widget)
        self.pushButton_2.setObjectName("pushButton_2")
        # self.pushButton_2.setFixedSize(100, 100)
        self.gridLayout.addWidget(self.pushButton_2, 0, 1, 1, 1)

        self.pushButton_3 = PrimaryPushButton(self.widget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 1, 0, 1, 1)

        self.pushButton_4 = PrimaryPushButton(self.widget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 1, 1, 1, 1)

        self.pushButton_5 = PrimaryPushButton(self.widget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 2, 0, 1, 1)

        self.pushButton_6 = PrimaryPushButton(self.widget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 2, 1, 1, 1)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, int(926*self.zoom), int(22*self.zoom)))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "燃烧我的卡路里！！！"))
        # self.label_2.setText(_translate("MainWindow", "人体姿态点信息"))
        self.pushButton_1.setText(_translate("MainWindow", "打开摄像头"))
        self.pushButton_2.setText(_translate("MainWindow", "停止运动"))
        self.pushButton_3.setText(_translate("MainWindow", "开始检测"))
        self.pushButton_4.setText(_translate("MainWindow", "开始运动"))
        self.pushButton_5.setText(_translate("MainWindow", "手机摄像头"))
        self.pushButton_6.setText(_translate("MainWindow", "PushButton6"))
        '''
        天依蓝 #66ccff
        初音绿 #66ffcc
        言和绿 #99ffff
        阿绫红 #ee0000
        双子黄 #ffff00
        红色：#FF0000
        绿色：#00FF00
        蓝色：#0000FF
        黄色：#FFFF00
        青色（Cyan）：#00FFFF
        品红（Magenta）：#FF00FF
        银色：#C0C0C0
        黑色：#000000
        白色：#FFFFFF
        灰色：#808080
        棕色：#A52A2A
        紫色：#800080
        橙色：#FFA500
        粉色：#FFC0CB
        金色：#FFD700
        深蓝：#00008B
        海军蓝：#000080
        天蓝：#ADD8E6
        淡蓝：#ADD8E6
        深绿：#006400
        橄榄绿：#808000
        碧绿：#00FF7F
        浅绿：#90EE90
        深红：#8B0000
        暗红：#8B008B
        玫瑰红：#FFC0CB
        '''