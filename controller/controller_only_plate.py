import os
import sys
import time
import cv2
import numpy as np
import imutils
import pytesseract
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from views.plate_detection import Ui_MainWindow
from utils import MoilUtils
from moilutils import mutils
from controller.videoController import VideoController
from datetime import datetime
from shapely.geometry import Polygon
from model.load_save import Load_save
from model.yolo_config import Yolo_config
from model.perspective_images import Perspective
from model.recognition import Recognition
from model.data_properties import DataProperties
from model.object_detection_functions import object_detection_analysis_with_nms


class controller(Ui_MainWindow):
    def __init__(self, MainWindow):
        super(Ui_MainWindow, self).__init__()
        self.parent = MainWindow
        self.setupUi(self.parent)
        self.title = "plate"
        self.Load_Save = Load_save(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame) # next_frame for video
        self.yolo_config = Yolo_config(self)
        self.recognition = Recognition(self)
        self.perspective = Perspective(self)
        self.moildev = None
        self.mutils = mutils
        self.data_properties = DataProperties()

        self.video_controller = VideoController(self)
        self.video = None
        self.cap = None
        self.w = 0
        self.h = 0
        self.camera = False
        self.image = None
        self.cam = False
        self.record = False
        self.video_writer = None
        self.videoDir = None
        self.num_in = 0
        self.num_out = 0

        self.anypoint_in = None
        self.anypoint_out = None
        self.maps_x_in = None
        self.maps_y_in = None
        self.maps_x_out = None
        self.maps_y_out = None

        self.Load_Save.load_param()
        self.connect()

    def connect(self):
        """
        This function is to connect each push button on the user interface with the every function in the controller
        """
        self.btn_open_file.clicked.connect(self.onclick_open_file)
        self.btn_open_video.clicked.connect(self.onclick_open_video)
        self.btn_open_camera.clicked.connect(self.onclick_open_camera)
        self.btn_cam_params.clicked.connect(self.onclick_params)
        self.btn_save_setting.clicked.connect(self.Load_Save.save_param)

        # self.btn_automatic.clicked.connect(self.load_automatic)
        self.btn_record.clicked.connect(self.save_to_record)
        self.btn_save_img_inside.clicked.connect(self.save_image_inside)
        self.btn_save_img_outside.clicked.connect(self.save_image_outside)

        # Operation for video player
        self.btn_play_pouse_3.clicked.connect(self.onclick_play_video)
        self.btn_prev_video_3.clicked.connect(self.onclick_prev_video)
        self.btn_stop_video_3.clicked.connect(self.onclick_stop_video)
        self.btn_skip_video_3.clicked.connect(self.onclick_skip_video)
        self.slider_Video_3.valueChanged.connect(self.onclick_slider_video)

        # Output for original image
        self.wind_image_source.mousePressEvent = self.mouse_event_image_source

        # Setting parameters for create anypoint maps inside
        self.val_alpha_in.valueChanged.connect(self.valueChange_inside)
        self.val_beta_in.valueChanged.connect(self.valueChange_inside)
        self.val_zoom_in.valueChanged.connect(self.valueChange_inside)

        # Setting parameters for create anypoint maps inside
        self.val_alpha_out.valueChanged.connect(self.valueChange_outside)
        self.val_beta_out.valueChanged.connect(self.valueChange_outside)
        self.val_zoom_out.valueChanged.connect(self.valueChange_outside)

    def onclick_open_file(self):
        filename = mutils.select_file(self.parent, "Select Image", "../sample_image/",
                                      "Image Files (*.jpeg *.jpg *.png *.gif *.bmg)")
        if filename:
            image = mutils.read_image(filename)
            mutils.show_image_to_label(self.wind_image_source, image, 400)

        else:
            QtWidgets.QMessageBox.information(self.parent, "Information", "No source camera founded")

    def onclick_open_video(self):
        video_source = mutils.select_file(self.parent,
                                          "Select Video Files",
                                          "../",
                                          "Video Files (*.mp4 *.avi *.mpg *.gif *.mov)")

        self.moildev_in = MoilUtils.connectToMoildev("entaniya")
        self.moildev_out = MoilUtils.connectToMoildev("entaniya")
        self.video = cv2.VideoCapture(video_source)

        self.valueChange_inside()
        self.valueChange_outside()
        self.next_frame()

    # def running_video(self):
    #     # self.cap = cv2.VideoCapture(self.video_source)
    #     self.next_frame()

    def next_frame(self):
        self.data_properties.properties_video["video"] = True
        if self.video:
            success, self.image = self.video.read()
            if success:
                start = time.time()
                self.timer.start()
                self.anypoint_inside()
                self.anypoint_outside()
                self.show_image()
                end = time.time()
                seconds = start - end
                print("time:{}".format(seconds))

    def get_value_slider_video(self, value):
        current_position = self.data_properties.properties_video["pos_frame"] * (value + 1) / \
                           self.data_properties.properties_video["frame_count"]
        return current_position

    def onclick_open_camera(self):
        """
        This function is to connected ip camera fisheye which will be use for record video capture images datasets
        (real time)
        """
        camera_link = "http://10.42.0.183:8000/stream.mjpg"
        self.video = cv2.VideoCapture(camera_link)

        self.moildev_in = MoilUtils.connectToMoildev("entaniya")
        self.moildev_out = MoilUtils.connectToMoildev("entaniya")

        self.valueChange_inside()
        self.valueChange_outside()
        self.next_frame()

    def onclick_params(self):
        """
        This function is for showing parameter (Create, Update, and Delete params)
        """
        cam_params = mutils.form_camera_parameter()
        print(cam_params)

    def next_frame_streaming(self):
        """
        This function is to continue video capture each frame current active (real time) while showing 2 part front and
        back from 1 fisheye image
        """
        if self.video:
            success, self.image = self.video.read()
            if success:
                self.data_properties.properties_video["video"] = True
                self.timer.start()
                self.anypoint_inside()
                self.anypoint_outside()
                self.show_image()

    def save_to_record(self):
        ret, image = self.video.read()
        h, w, z = image.shape

        record = True
        print("Record video")
        out = []
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out.append(cv2.VideoWriter("./Videos/output_video.avi", fourcc, 10, (w, h)))

        while self.video.isOpened():
            success, frame = self.video.read()
            if success:
                cv2.imwrite("./images/right_true_park5.jpg", frame)
                if record:
                    out[0].write(frame)

    def valueChange_inside(self):
        """
        This function is to activ the value to create anypoint maps inside with setting alpha, beta, and zoom as the
        parameter
        """
        alpha = self.val_alpha_in.value()
        beta = self.val_beta_in.value()
        zoom = self.val_zoom_in.value()
        self.maps_x_in, self.maps_y_in = self.moildev_in.getAnypointMaps(alpha, beta, zoom)
        self.x_in, self.y_in, self.center_fish, self.width, self.height = self.CenterGravity(self.maps_x_in, self.maps_y_in)

    def anypoint_inside(self):
        """
        This function is implement anypoint inside map with the method detector from openCV (Yolo) to detect all object
        in area fisheye camera and make the object position become to center which will be show on the user interface
        """
        self.anypoint_in = MoilUtils.remap(self.image, self.maps_x_in, self.maps_y_in)
        cv2.imwrite("./images/anypoint_inside/image.jpg", self.anypoint_in)
        self.anypoint_in_draw, self.position_in, self.prediction_confidence = self.yolo_config.detect_using_yolo(self.anypoint_in)
        # self.plate_recognition_in = self.perspective.perspective_images(self.anypoint_in_draw)
        if self.prediction_confidence != 0:
            self.num_in = self.num_in+1

        # self.plat_num = self.recognition.recognition_character(self.anypoint_in)
        # self.plate_recognition_in.setText(str(self.plat_num))
        self.num_detect_in.setText(str(self.num_in))
        print(type(self.prediction_confidence))
        self.prediction_in.setText(str(round(self.prediction_confidence * 100, 2)) + " %")

        if self.position_in is not None:
            frame_in = self.crop_image_detected(self.anypoint_in, self.position_in)
            print(frame_in)
            center_position_obj = self.position_in[0][0] + self.position_in[0][2]/2, self.position_in[0][1] + \
                                  self.position_in[0][3]/2
            # print(center_position_obj)
            print(self.center_fish, self.width, self.height)

            real_distance_w = self.width * center_position_obj[0] / self.anypoint_in.shape[1]
            real_distance_h = self.height * center_position_obj[1] / self.anypoint_in.shape[0]
            # delta_center_x = self.anypoint_in.shape[1] - center_position[0]
            # delta_center_y = self.anypoint_in.shape[0] - center_position[1]
            # if delta_center_y

            cv2.circle(self.anypoint_in_draw, (int(center_position_obj[0]), int(center_position_obj[1])),
                       15, (0, 0, 255), -1)
            cv2.circle(self.image,
                       (int(self.center_fish[0][0]), int(self.center_fish[0][1])),
                       15, (0, 0, 255), -1)
            # cv2.circle(self.image,
            #            (int(self.x_in+real_distance_w), int(self.y_in+real_distance_h)),
            #            10, (0, 0, 255), -1)

            self.createAlphaBeta(int(self.x_in+real_distance_w), int(self.y_in+real_distance_h))
            # if center_position[0] >= self.anypoint_in.shape[1]
            MoilUtils.showImageToLabel(self.wind_detected_in, frame_in, 300)
            cv2.imwrite("./images/detected_inside/image.jpg", frame_in)
            self.pix_num_image_in.setText(str(self.center_fish))

        else:
            self.wind_detected_in.setText("No Detected")
            self.wind_detected_in.setText("No Image")

        # if self.position_in is not None:
        MoilUtils.showImageToLabel(self.wind_inside_image, self.anypoint_in_draw, 700)

    def anypoint_outside(self):
        """
        This function is implement anypoint inside map with the method detector dari openCV (Yolo) to detect all object
        in area fisheye camera
        """
        self.anypoint_out = MoilUtils.remap(self.image, self.maps_x_out, self.maps_y_out)
        # cv2.imwrite("./images/anypoint_outside/image.jpg", self.anypoint_out)
        self.anypoint_out_draw, self.position_out, self.prediction_confidence = self.yolo_config.detect_using_yolo(self.anypoint_out)
        # if self.position_out is not None:
        #     frame = self.crop_image_detected(self.anypoint_out, self.position_out)
        #     MoilUtils.showImageToLabel(self.wind_detected_in, frame, 200)
        # else:
        #     self.wind_detected_in.setText
        if self.prediction_confidence != 0:
            self.num_in = self.num_in + 1

        # self.plat_num = self.recognition.recognition_character(self.anypoint_in)
        # self.plate_recognition_in.setText(str(self.plat_num))
        self.num_detect_out.setText(str(self.num_out))
        print(type(self.prediction_confidence))
        self.prediction_out.setText(str(round(self.prediction_confidence * 100, 2)) + " %")

        if self.position_out is not None:
            frame_out = self.crop_image_detected(self.anypoint_out, self.position_out)
            center_position_obj = self.position_out[0][0] + self.position_out[0][2]/2, self.position_out[0][1] + \
                                  self.position_out[0][3]/2
            # print(center_position_obj)
            print(self.center_fish, self.width, self.height)

            real_distance_w = self.width * center_position_obj[0] / self.anypoint_in.shape[1]
            real_distance_h = self.height * center_position_obj[1] / self.anypoint_in.shape[0]
            cv2.circle(self.anypoint_out_draw, (int(center_position_obj[0]), int(center_position_obj[1])),
                       15, (255, 0, 255), -1)
            cv2.circle(self.image,
                       (int(self.center_fish[0][0]), int(self.center_fish[0][1])),
                       15, (255, 0, 255), -1)
            # cv2.circle(self.image,
            #            (int(self.x_in+real_distance_w), int(self.y_in+real_distance_h)),
            #            63, (0, 0, 255), -1)

            self.createAlphaBeta_out(int(self.x_in+real_distance_w), int(self.y_in+real_distance_h))
            MoilUtils.showImageToLabel(self.wind_detected_out, frame_out, 300)
            # cv2.imwrite("./images/detected_outside/image.jpg", frame_out)
            self.pix_num_image_out.setText(str(self.center_fish))

        else:
            self.wind_detected_out.setText("No Detected")
            self.wind_detected_out.setText("No Image")

        MoilUtils.showImageToLabel(self.wind_outsid_image, self.anypoint_out_draw, 700)

    def createAlphaBeta(self, x, y):
        """
        This function is to create alpha and beta anypoint using moilutils remap (rectilinier) method from our labs
        """
        alpha, beta = self.moildev_in.getAlphaBeta(x, y, mode=2)
        self.moildev_new = MoilUtils.connectToMoildev("entaniya")
        mapsx, mapsy = self.moildev_new.getAnypointMaps(alpha, beta, zoom=20)
        anypoint = MoilUtils.remap(self.image, mapsx, mapsy)
        MoilUtils.showImageToLabel(self.wind_detected_in, anypoint, 200)
        return anypoint

    def createAlphaBeta_out(self, x, y):
        """
        This function is to create alpha and beta anypoint using moilutils remap (rectilinier) method from our labs
        """
        alpha, beta = self.moildev_out.getAlphaBeta(x, y, mode=2)
        self.moildev_new_out = MoilUtils.connectToMoildev("entaniya")
        mapsx, mapsy = self.moildev_new_out.getAnypointMaps(alpha, beta, zoom=20)
        anypoint_out = MoilUtils.remap(self.image, mapsx, mapsy)
        MoilUtils.showImageToLabel(self.wind_detected_out, anypoint_out, 200)
        return anypoint_out

    @classmethod
    def CenterGravity(cls, maps_x, maps_y):
        a = [maps_x[0][0], maps_y[0][0]]
        b = [maps_x[0][-1], maps_y[0][-1]]
        c = [maps_x[-1][-1], maps_y[-1][-1]]
        d = [maps_x[-1][0], maps_y[-1][0]]

        width = maps_x[0][-1] - maps_x[0][0]
        height = maps_y[-1][0] - maps_y[0][0]

        x = maps_x[0][0]
        y = maps_y[0][0]

        center = list(Polygon([a, b, c, d]).centroid.coords)
        return x, y, center, width, height

    @classmethod
    def crop_image_detected(cls, frame, box):
        frame2 = frame[box[0][1]:box[0][1] + box[0][3],box[0][0]:box[0][0] + box[0][2]]

        return frame2

    def show_image(self):
        """
        This function to showing two rectilinear view image inside (front) and fisheye image outside (back)
        from fisheye image
        """
        image_draw = self.image.copy()
        image_draw = MoilUtils.drawPolygon(image_draw, self.maps_x_in, self.maps_y_in)
        image_draw = MoilUtils.drawPolygon(image_draw, self.maps_x_out, self.maps_y_out)
        MoilUtils.showImageToLabel(self.wind_image_source, image_draw, 400)

    def save_image_inside(self):
        """
        This function is to save image anypoint inside (front) with the format time now
        """
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        cv2.imwrite("./images/anypoint_inside/img_in_" + curr_datetime + ".jpg", self.anypoint_in)
        print("test")

    def save_image_outside(self):
        """
        This function is to save image anypoint outside (back) with the format time now
        """
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        cv2.imwrite("./images/anypoint_outside/img_out_" + curr_datetime + ".jpg", self.anypoint_out)
        print("tesss")

    def valueChange_outside(self):
        """
        This function is to change the value parameters (alpha, beta, and zoom) for got the anypoint map (outside)
        :return:
        """
        print("value change outside")
        alpha = self.val_alpha_out.value()
        beta = self.val_beta_out.value()
        zoom = self.val_zoom_out.value()
        print("outside = ", alpha, beta, zoom)
        self.maps_x_out, self.maps_y_out = self.moildev_out.getAnypointMaps(alpha, beta, zoom)

    def setting(self):
        """
        This function is setting camera which be use
        """
        if self.pushButton_3.isChecked():
            self.frame_setting_camera.hide()
        else:
            self.frame_setting_camera.show()

    def mouse_event_image_source(self, e):
        """
        This function is to select area for new icx and icy on image
        """
        print("click even")
        if e.button() == Qt.MouseButton.LeftButton:
            pos_x = round(e.position().x())
            pos_y = round(e.position().y())
            if self.image is None:
                print("no image")
            else:
                ratio_x, ratio_y = MoilUtils.ratio_image_to_label(self.wind_image_source, self.image)
                icx_front = round(pos_x * ratio_x)
                icy_front = round(pos_y * ratio_y)

                alpha, beta = self.moildev_in.getAlphaBeta(icx_front, icy_front)
                if self.radioButton_inside.isChecked():
                    print("inside")
                    self.blockSignals()
                    self.val_alpha_in.setValue(alpha)
                    self.val_beta_in.setValue(beta)
                    self.unblockSignals()
                    self.valueChange_inside()
                if self.radioButton_outside.isChecked():
                    self.blockSignals()
                    print("outside")
                    self.val_alpha_out.setValue(alpha)
                    self.val_beta_out.setValue(beta)
                    self.unblockSignals()
                    self.valueChange_outside()

        elif e.button() == Qt.MouseButton.RightButton:
            if self.image is not None:
                menu = QtWidgets.QMenu()
                save = menu.addAction("Open Image")
                info = menu.addAction("Show Info")
                save.triggered.connect(self.onclick_open_camera)
                # info.triggered.connect(self.onclick_help)
                menu.exec(e.globalPos())

    def onclick_play_video(self):
        if self.image is not None:
            if self.btn_play_pouse_3.isChecked():
                self.timer.start()
            else:
                self.timer.stop()

    def onclick_prev_video(self):
        self.btn_prev_video_3.setChecked(False)
        self.rewind_video()
        self.timer.stop()

    def onclick_stop_video(self):
        self.btn_play_pouse_3.setChecked(False)
        self.stop_video()
        self.timer.stop()

    def onclick_skip_video(self):
        self.btn_skip_video_3.setChecked(False)
        self.forward_video()
        self.timer.stop()

    def rewind_video(self):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        position = self.data_properties.properties_video["pos_frame"] - 5 * fps
        self.video.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.next_frame()
        print("test rewind")

    def stop_video(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.next_frame()

    def forward_video(self):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        position = self.data_properties.properties_video["pos_frame"] + 5 * fps
        self.video.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.next_frame()
        print("test forward")

    def onclick_slider_video(self, value):
        value_max = self.slider_Video_3.maximum()
        self.slider_controller(value, value_max)

    def slider_controller(self, value, slider_maximum):
        dst = self.data_properties.properties_video["frame_count"] * value / slider_maximum
        self.video.set(cv2.CAP_PROP_POS_FRAMES, dst)
        self.next_frame()


    def blockSignals(self):
        """
        This function is for activate sphinx box on the user interface
        """
        self.val_alpha_in.blockSignals(True)
        self.val_beta_in.blockSignals(True)
        self.val_zoom_in.blockSignals(True)
        self.val_alpha_out.blockSignals(True)
        self.val_beta_out.blockSignals(True)
        self.val_zoom_out.blockSignals(True)

    def unblockSignals(self):
        """
        This function is for activate sphinx box on the user interface
        """
        self.val_alpha_in.blockSignals(False)
        self.val_beta_in.blockSignals(False)
        self.val_zoom_in.blockSignals(False)
        self.val_alpha_out.blockSignals(False)
        self.val_beta_out.blockSignals(False)
        self.val_zoom_out.blockSignals(False)


