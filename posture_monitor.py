import mediapipe
import PyQt5
from posture_qt import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLineEdit
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
import mediapipe as mp
import cv2

import matplotlib

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np

import math


class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # self.fig = plt.figure
        # 第二步：在父类中激活Figure窗口
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111, projection="3d")
        # self.axes = None
    # 第四步：就是画图，【可以在此类中画，也可以在其它类中画】


def cal_angle_3D(point_a, point_b, point_c):
    # a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
    # a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标
    # a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # 点a、b、c的z坐标
    # x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
    # x2, y2, z2 = (c_x - b_x), (c_y - b_y), (c_z - b_z)
    # cos_b = (x1 * x2 + y1 * y2 + z1 * z2) / (
    #         math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)))  # 角点b的夹角余弦值
    # B = math.degrees(math.acos(cos_b))  # 角点b的夹角值

    p1, p2, p3 = np.array(point_a), np.array(point_b), np.array(point_c)
    v1, v2 = p1 - p2, p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)
    return angle


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # UI界面
        self.setupUi(self)
        self.CAM_NUM = 0
        # self.cap = cv2.VideoCapture()
        # 在label中播放视频
        self.init()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.figure_draw = MyFigure()
        self.open_motion = False
        # self.groupBox = QtWidgets.QGroupBox(self.plt3d_module)
        # self.groupBox.setMinimumSize(QSize(1100, 610))
        # self.groupBox.setTitle("画图demo")
        # self.glo_plt_figure = QtWidgets.QGridLayout(self.groupBox)

    def init(self):
        # 定时器让其定时读取显示图片
        self.camera_timer = QTimer()
        self.read_img = QTimer()
        self.start_exercise = QTimer()

        self.camera_timer.timeout.connect(self.show_image)
        self.read_img.timeout.connect(self.posture_mediapipe_writer_timer)
        self.start_exercise.timeout.connect(self.motion_detection_timer)
        # self.camera_timer2.timeout.connect(self.open_vedio)
        # 打开摄像头
        self.pushButton_1.clicked.connect(self.open_camera)
        # 关闭摄像头
        self.pushButton_2.clicked.connect(self.stop_motion)
        # 测人体
        self.pushButton_3.clicked.connect(self.posture_mediapipe)
        # 开始运动
        self.pushButton_4.clicked.connect(self.motion_detection)

        self.pushButton_1.setEnabled(True)
        # 初始状态不能关闭摄像头
        self.pushButton_2.setEnabled(False)

    def open_camera(self):
        # self.cap = cv2.VideoCapture("5njf7-yoqfb.avi")  # 摄像头
        self.cap = cv2.VideoCapture(0)

        self.label.setScaledContents(True)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 930)
        self.camera_timer.start(40)  # 每40毫秒读取一次，即刷新率为25帧
        self.show_image()

    def show_image(self):
        flag, self.image = self.cap.read()  # 从视频流中读取图片
        self.image = cv2.flip(self.image, 1)
        if not flag:
            return
        self.width, self.height = self.image.shape[:2]  # 行:宽，列:高
        self.results = self.holistic.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.posture_mediapipe_draw()
        if self.open_motion:
            cv2.putText(self.image, "counter{}".format(self.motion_detection_counter), (400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), thickness=3)
        self.showImage = QtGui.QImage(self.image.data, self.height, self.width, QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.showImage))  # 往显示视频的Label里显示QImage

    def posture_mediapipe_draw(self):
        # self.mp_drawing.draw_landmarks(self.image,
        #                                self.results.face_landmarks,
        #                                self.mp_holistic.FACEMESH_CONTOURS,
        #                                landmark_drawing_spec=None,
        #                                connection_drawing_spec=self.mp_drawing_styles
        #                                .get_default_face_mesh_contours_style())

        self.mp_drawing.draw_landmarks(self.image,
                                       self.results.pose_landmarks,
                                       self.mp_holistic.POSE_CONNECTIONS,
                                       landmark_drawing_spec=self.mp_drawing_styles
                                       .get_default_pose_landmarks_style())

        self.mp_drawing.draw_landmarks(self.image, self.results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(self.image, self.results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    # 希望下辈子不再写屎山
    def posture_mediapipe(self):
        # self.results = self.holistic.process(self.image)
        # self.image.flags.writeable = True
        colorclass = plt.cm.ScalarMappable(cmap='jet')
        colors = colorclass.to_rgba(np.linspace(0, 1, int(33)))
        self.colormap = (colors[:, 0:3])
        # self.figure_draw.axes_3d = self.figure_draw.fig.add_subplot(111, projection='3d')
        self.gridLayout_plt.addWidget(self.figure_draw)  # 此句有bug
        self.read_img.start(10)
        self.posture_mediapipe_writer_timer()

    def posture_mediapipe_writer_timer(self):
        self.figure_draw.axes.clear()
        self.figure_draw.axes.set_xlim3d(-0.5, 1.5)
        self.figure_draw.axes.set_ylim3d(-1.5, 0.5)
        self.figure_draw.axes.set_zlim3d(-3, 1)
        # self.figure_draw.axes.set_xlim3d(-1, 1)
        # self.figure_draw.axes.set_ylim3d(-1, 1)
        # self.figure_draw.axes.set_zlim3d(-1, 1)
        if self.results.pose_landmarks:
            landmarks = []
            for index, landmark in enumerate(self.results.pose_landmarks.landmark):
                landmarks.append([landmark.x, landmark.z, landmark.y * (-1)])
            landmarks = np.array(landmarks)
            # print(landmarks)
            self.figure_draw.axes.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c=np.array(self.colormap),
                                          s=50)
            for _c in mp.solutions.pose.POSE_CONNECTIONS:
                self.figure_draw.axes.plot([landmarks[_c[0], 0], landmarks[_c[1], 0]],
                                           [landmarks[_c[0], 1], landmarks[_c[1], 1]],
                                           [landmarks[_c[0], 2], landmarks[_c[1], 2]], 'k')
            # self.gridLayout_plt.addWidget(self.figure_draw)
            # plt.pause(0.001)
            self.figure_draw.draw()

    def motion_detection(self):
        self.motion_detection_flag = True
        self.open_motion = True
        self.motion_detection_counter = 0
        items = ("举哑铃", "高抬腿", "下蹲")
        self.item, self.chosen_spot = \
            QtWidgets.QInputDialog.getItem(self, "选择做哪个运动呢", "运动选择:", items, 0, False)
        print(self.item, self.chosen_spot)
        self.pushButton_2.setEnabled(True)
        self.start_exercise.start(40)
        self.motion_detection_timer()

    def motion_detection_timer(self):
        if self.results.pose_landmarks:
            right_shoulder = [self.results.pose_landmarks.landmark[11].x * self.width,
                              self.results.pose_landmarks.landmark[11].y * self.height,
                              self.results.pose_landmarks.landmark[11].z]
            right_elbow = [self.results.pose_landmarks.landmark[13].x * self.width,
                           self.results.pose_landmarks.landmark[13].y * self.height,
                           self.results.pose_landmarks.landmark[13].z]
            right_wrist = [self.results.pose_landmarks.landmark[15].x * self.width,
                           self.results.pose_landmarks.landmark[15].y * self.height,
                           self.results.pose_landmarks.landmark[15].z]
            # print(right_shoulder, right_elbow, right_wrist)
            arm_angle = cal_angle_3D(right_shoulder, right_elbow, right_wrist)
            print(arm_angle, self.motion_detection_flag)
            if arm_angle <= 40 and self.motion_detection_flag:
                self.motion_detection_counter = self.motion_detection_counter + 1
                self.motion_detection_flag = False
            if arm_angle >= 150 and not self.motion_detection_flag:
                self.motion_detection_flag = True

    # 停止运动
    def stop_motion(self):
        # self.label.clear()
        self.pushButton_2.setEnabled(False)
        # self.timer.stop()
        # self.read_img.stop()
        # self.camera_timer.stop()
        self.start_exercise.stop()
        self.open_motion = False
        self.motion_detection_counter = 0

    # 播放视频画面
    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_pic)

    # # 显示视频图像
    # def show_pic(self):
    #     ret, img = self.cap.read()
    #     if ret:
    #         cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         # 视频流的长和宽
    #         height, width = cur_frame.shape[:2]
    #         pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
    #         pixmap = QPixmap.fromImage(pixmap)
    #         # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
    #         ratio = max(width / self.label.width(), height / self.label.height())
    #         pixmap.setDevicePixelRatio(ratio)
    #         # 视频流置于label中间部分播放
    #         self.label.setAlignment(Qt.AlignCenter)
    #         self.label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
