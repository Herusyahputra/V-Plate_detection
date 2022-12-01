import time
import PIL.ImageGrab
import cv2
import numpy as np
import imutils
import pytesseract
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from views.plate_detection import Ui_MainWindow
from utils import MoilUtils
from controller.videoController import VideoController
from datetime import datetime
from shapely.geometry import Polygon
from model.load_save import Load_save
from model.yolo_config import Yolo_config
from model.perspective_images import Perspective
from model.recognition import Recognition
from model.data_properties import DataProperties

# Start from Anto teach style
class controller(Ui_MainWindow):
    def __init__(self, MainWindow):
        super(Ui_MainWindow, self).__init__()
        self.parent = MainWindow
        self.setupUi(self.parent)
        self.title = "plate"
        self.Load_Save = Load_save(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.yolo_config = Yolo_config(self)
        self.recognition = Recognition(self)
        self.perspective = Perspective(self)
        self.moildev = None
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
        self.num_ori = 0
        self.num_in = 0
        self.num_out = 0

        self.original_image = None
        self.anypoint_in = None
        self.anypoint_out = None
        self.maps_x_in = None
        self.maps_y_in = None
        self.maps_x_out = None
        self.maps_y_out = None

        self.point_in = []
        self.point_out = []
        self.image_click_plate = None
        self.image_click_plate_zone2 = None

        self.val_alpha_in.valueChanged.connect(self.valueChange_inside)
        self.Load_Save.load_param()
        self.connect()

    def connect(self):
        """
        This function is to connect each push button on the user interface with the every function in the controller
        """
        self.btn_open_file.clicked.connect(self.onclick_open_file)
        self.btn_open_video.clicked.connect(self.onclick_open_video)
        self.btn_open_camera.clicked.connect(self.onclick_open_camera)
        self.btn_record.clicked.connect(self.save_to_record)
        self.btn_save_img_inside.clicked.connect(self.save_image_inside)
        self.btn_save_img_outside.clicked.connect(self.save_image_outside)
        self.btn_save_setting.clicked.connect(self.Load_Save.save_param)

        # Operation for video player
        self.btn_play_pouse_3.clicked.connect(self.onclick_play_video)
        self.btn_prev_video_3.clicked.connect(self.onclick_prev_video)
        self.btn_stop_video_3.clicked.connect(self.onclick_stop_video)
        self.btn_skip_video_3.clicked.connect(self.onclick_skip_video)
        self.slider_Video_3.valueChanged.connect(self.onclick_slider_video)

        # Output for original image
        self.original_fisheye.mousePressEvent = self.mouse_event_image_source
        self.wind_inside_image.mousePressEvent = self.mouse_event_image_anypoint
        self.wind_outsid_image.mousePressEvent = self.mouse_event_image_anypoint_out

        # Setting parameters for create anypoint maps inside
        self.val_alpha_in.valueChanged.connect(self.valueChange_inside)
        self.val_beta_in.valueChanged.connect(self.valueChange_inside)
        self.val_zoom_in.valueChanged.connect(self.valueChange_inside)
        self.rotate_in.valueChanged.connect(self.anypoint_zone_1)
        self.rotate_out.valueChanged.connect(self.anypoint_zone_2)

        self.mode_1_in.toggled.connect(self.valueChange_inside)
        self.mode_2_in.toggled.connect(self.valueChange_inside)
        self.mode_1_out.toggled.connect(self.valueChange_outside)
        self.mode_2_out.toggled.connect(self.valueChange_outside)

        # Setting parameters for create anypoint maps outside
        self.val_alpha_out.valueChanged.connect(self.valueChange_outside)
        self.val_beta_out.valueChanged.connect(self.valueChange_outside)
        self.val_zoom_out.valueChanged.connect(self.valueChange_outside)

        self.rotate_ori.valueChanged.connect(self.normal_fisheye)

    # Start from MoilUtils-Template
    def onclick_open_file(self):
        """
        This function is useful when the user opens an image file that will be processed, a file format like, (*.jpeg *.jpg *.png *.gif *.bmg)
        """
        filename = MoilUtils.selectFile(self.parent, "Select Image", "./sample_image/",
                                        "Image Files (*.jpeg *.jpg *.png *.gif *.bmg)")

        self.moildev_in = MoilUtils.connectToMoildev("entaniya")
        self.moildev_out = MoilUtils.connectToMoildev("entaniya")
        self.image = MoilUtils.readImage(filename)
        self.valueChange_inside()
        self.valueChange_outside()
        self.normal_fisheye()
        self.anypoint_zone_1()
        self.anypoint_zone_2()
        self.next_frame()
        self.show_image()

    def onclick_open_video(self):
        """
        This function is useful when the user opens an video file that will be processed, a file format like, (*.mp4 *.avi *.mpg *.gif *.mov)
        """
        video_source = MoilUtils.selectFile(self.parent,
                                            "Select Video Files",
                                            "../",
                                            "Video Files (*.mp4 *.avi *.mpg *.gif *.mov)")

        self.moildev_in = MoilUtils.connectToMoildev("entaniya")
        self.moildev_out = MoilUtils.connectToMoildev("entaniya")
        self.video = cv2.VideoCapture(video_source)

        self.valueChange_inside()
        self.valueChange_outside()
        self.next_frame()

    def next_frame(self):
        """This function will be executed when the user runs a video associated with several functions for processing
        images and videos with endless loops
        """
        self.data_properties.properties_video["video"] = True
        if self.video:
            success, self.image = self.video.read()
            if success:
                start = time.time()
                self.timer.start()
                self.normal_fisheye()
                self.anypoint_zone_1()
                self.anypoint_zone_2()
                self.show_image()
                print("process streaming")
                end = time.time()
                seconds = end - start
                print("time:{}".format(seconds))

        # start Anto guide source-code
        else:
            start = time.time()
            self.timer.start()
            self.show_image()
            print("process image")
            end = time.time()
            seconds = end - start
            print("image mode, time:{}".format(seconds))

        # end Anto guide source-code

    def get_value_slider_video(self, value):
        """
        This function is useful for viewing the time when running a video on the application
        """
        current_position = self.data_properties.properties_video["pos_frame"] * (value + 1) / \
                           self.data_properties.properties_video["frame_count"]
        return current_position

    def onclick_open_camera(self):
        """
        This function will open the camera with the ip link camera used, when the user will stream with
        real time video when the application process is running
        """
        print("test streaming")
        camera_link = "http://10.42.0.170:8000/stream.mjpg"
        self.video = cv2.VideoCapture(camera_link)

        self.moildev_in = MoilUtils.connectToMoildev("entaniya")
        self.moildev_out = MoilUtils.connectToMoildev("entaniya")

        self.valueChange_inside()
        self.valueChange_outside()
        self.next_frame()

    def next_frame_streaming(self):
        """
        This function will be used for live-streaming and calling some functions to process images
        """
        if self.video:
            success, self.image = self.video.read()
            if success:
                self.data_properties.properties_video["video"] = True
                self.timer.start()
                self.normal_fisheye()
                self.anypoint_zone_1()
                self.anypoint_zone_2()
                self.show_image()
    # End from MoilUtils-Template

    def save_to_record(self):
        """
        This function will be used when the user clicks the recording button to record images in video form
        format *.avi
        """
        ret, image = self.video.read()
        h, w, z = image.shape

        record = True
        print("Recording")
        out = []
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out.append(cv2.VideoWriter("./Videos/output_video.avi", fourcc, 10, (w, h)))

        while self.video.isOpened():
            success, frame = self.video.read()
            if success:
                cv2.imwrite("./images/original_image.jpg", frame)
                if record:
                    out[0].write(frame)

    def normal_fisheye(self):
        """
        This function will be useful for processing original images, by detecting objects in images using
        Yolo algorithm which will be displayed in the user interface
        """
        print("normal image")
        self.ori_fisheye = self.image
        self.ori_fisheye = self.rotate_value_ori(self.ori_fisheye)
        cv2.imwrite("./images/normal_fisheye.jpg", self.ori_fisheye)
        self.anypoint_ori_draw, self.position_ori, self.prediction_confidence = self.yolo_config.detect_using_yolo(
            self.ori_fisheye)
        if self.prediction_confidence != 0:
            self.num_ori = self.num_ori + 1

        self.num_detect_ori.setText(str(self.num_ori))
        self.prediction_ori.setText(str(round(self.prediction_confidence * 100, 2)) + " %")

        if self.position_ori is not None:
            frame_ori = self.crop_image_detected(self.ori_fisheye, self.position_ori)
            center_position_obj = self.position_ori[0][0] + self.position_ori[0][2] / 2, self.position_ori[0][1] + \
                                  self.position_ori[0][3] / 2
            print(self.center_fish, self.width, self.height)

            real_distance_w = self.width * center_position_obj[0] / self.ori_fisheye.shape[1]
            real_distance_h = self.height * center_position_obj[1] / self.ori_fisheye.shape[0]

            # cv2.circle(self.anypoint_ori_draw, (int(center_position_obj[0]), int(center_position_obj[1])),
            #            15, (0, 0, 255), -1)
            # cv2.circle(self.image,
            #            (int(self.center_fish[0][0]), int(self.center_fish[0][1])),
            #            15, (0, 0, 255), -1)

            # self.createAlphaBeta(int(self.x_in + real_distance_w), int(self.y_in + real_distance_h))
            MoilUtils.showImageToLabel(self.repository, frame_ori, 300)

        else:
            self.repository.setText("No detected")

        MoilUtils.showImageToLabel(self.original_fisheye, self.anypoint_ori_draw, 600)

    # Start from Moildev-SDK
    def anypoint_zone_1(self):
        """
        This function will be used to create a new Anypoint in Zone 1 from the original image which can detect
        objects automatically using the Yolo algorithm and perspective transform as a manual method
        """
        print("anypoint zone 1")
        # self.anypoint_in = self.image
        self.anypoint_in = MoilUtils.remap(self.image, self.maps_x_in, self.maps_y_in)
        self.anypoint_in = self.rotate_value_in(self.anypoint_in)
        # print("rotate in: {}". format(self.anypoint_in))
        cv2.imwrite("./images/anypoint_inside/image.jpg", self.anypoint_in)
        self.anypoint_in_draw, self.position_in, self.prediction_confidence = self.yolo_config.detect_using_yolo(
            self.anypoint_in)

        if self.prediction_confidence != 0:
            self.num_in = self.num_in + 1

        self.num_detect_in.setText(str(self.num_in))
        # self.num_detect_ori.setText(str(self.num_in))
        print(type(self.prediction_confidence))

        # recognition
        # self.plat_num = self.recognition.recognition_character(self.anypoint_in)
        # self.num_detect_ori.setText(str(self.plat_num))
        self.prediction_in.setText(str(round(self.prediction_confidence * 100, 2)) + " %")
        # self.prediction_ori.setText(str(round(self.prediction_confidence * 100, 2)) + " %")

        if self.position_in is not None:
            frame_in = self.crop_image_detected(self.anypoint_in, self.position_in)
            center_position_obj = self.position_in[0][0] + self.position_in[0][2] / 2, self.position_in[0][1] + \
                                  self.position_in[0][3] / 2
            print(self.center_fish, self.width, self.height)

            real_distance_w = self.width * center_position_obj[0] / self.anypoint_in.shape[1]
            real_distance_h = self.height * center_position_obj[1] / self.anypoint_in.shape[0]
            cv2.circle(self.anypoint_in_draw, (int(center_position_obj[0]), int(center_position_obj[1])),
                       15, (0, 0, 255), -1)
            # cv2.circle(self.image,
            #            (int(self.center_fish[0][0]), int(self.center_fish[0][1])),
            #            15, (0, 0, 255), -1)
            # cv2.circle(self.image,
            #            (int(self.x_in+real_distance_w), int(self.y_in+real_distance_h)),
            #            10, (0, 0, 255), -1)

            self.createAlphaBeta(int(self.x_in + real_distance_w), int(self.y_in + real_distance_h))
            MoilUtils.showImageToLabel(self.wind_detected_in, frame_in, 200)
            # MoilUtils.showImageToLabel(self.repository, frame_in, 200)
            cv2.imwrite("./images/detected_inside/image.jpg", frame_in)
            # self.pix_num_image_in.setText(str(self.center_fish))

        else:
            self.wind_detected_in.setText("No Detected")
            self.wind_detected_in.setText("No Image")
        self.show_image_anypoint_draw()

    def show_image_anypoint_draw(self):
        """
        This function will be used to display the result of creating a new Anypoint in Zone 1
        """
        # if self.position_in is not None:
        # MoilUtils.showImageToLabel(self.original_fisheye, self.anypoint_in_draw, 600)
        MoilUtils.showImageToLabel(self.wind_inside_image, self.anypoint_in_draw, 600)
        if self.image_click_plate is not None:
            MoilUtils.showImageToLabel(self.wind_detected_in_m, self.image_click_plate, 200)

    def anypoint_zone_2(self):
        """
        This function will be used to create a new Anypoint in Zone 2 from the original image which can detect
        objects automatically using the Yolo algorithm and perspective transform as a manual method
        """
        print("anypoint zone 2")
        # self.anypoint_out = self.image
        self.anypoint_out = MoilUtils.remap(self.image, self.maps_x_out, self.maps_y_out)
        self.anypoint_out = self.rotate_value_out(self.anypoint_out)
        cv2.imwrite("./images/anypoint_outside/image.jpg", self.anypoint_out)
        self.anypoint_out_draw, self.position_out, self.prediction_confidence = self.yolo_config.detect_using_yolo(
            self.anypoint_out)
        if self.prediction_confidence != 0:
            self.num_out = self.num_out + 1

        self.num_detect_out.setText(str(self.num_out))
        print(type(self.prediction_confidence))

        # recognition
        # self.plat_num = self.recognition.recognition_character(self.anypoint_out)
        # self.num_detect_ori.setText(str(self.plat_num))
        self.prediction_out.setText(str(round(self.prediction_confidence * 100, 2)) + " %")

        if self.position_out is not None:
            frame_out = self.crop_image_detected(self.anypoint_out, self.position_out)
            center_position_obj = self.position_out[0][0] + self.position_out[0][2] / 2, self.position_out[0][1] + \
                                  self.position_out[0][3] / 2
            print(self.center_fish, self.width, self.height)

            real_distance_w = self.width * center_position_obj[0] / self.anypoint_out.shape[1]
            real_distance_h = self.height * center_position_obj[1] / self.anypoint_out.shape[0]
            cv2.circle(self.anypoint_out_draw, (int(center_position_obj[0]), int(center_position_obj[1])),
                       15, (255, 0, 255), -1)
            # cv2.circle(self.image,
            #            (int(self.center_fish[0][0]), int(self.center_fish[0][1])),
            #            15, (255, 0, 255), -1)
            # cv2.circle(self.image,
            #            (int(self.x_in+real_distance_w), int(self.y_in+real_distance_h)),
            #            63, (0, 0, 255), -1)

            self.createAlphaBeta_out(int(self.x_out + real_distance_w), int(self.y_out + real_distance_h))
            MoilUtils.showImageToLabel(self.wind_detected_out, frame_out, 200)
            cv2.imwrite("./images/detected_outside/image.jpg", frame_out)
            # self.pix_num_image_out.setText(str(self.center_fish))

        else:
            self.wind_detected_out.setText("No Detected")
            self.wind_detected_out.setText("No Image")

        self.show_image_anypoint_draw_out()
    # End Moildev-SDK

    def show_image_anypoint_draw_out(self):
        """
        This function will be used to display the result of creating a new Anypoint in Zone 2
        """
        MoilUtils.showImageToLabel(self.wind_outsid_image, self.anypoint_out_draw, 600)
        if self.image_click_plate_zone2 is not None:
            MoilUtils.showImageToLabel(self.wind_detected_out_m, self.image_click_plate_zone2, 200)

    def createAlphaBeta(self, x, y):
        """
        This function is useful for making alpha, beta in images on zone 1
        """
        alpha, beta, = self.moildev_in.get_alpha_beta(x, y, mode=2)
        mapsx, mapsy = self.moildev_in.maps_anypoint(alpha, beta, zoom=20, mode=2)
        anypoint = MoilUtils.remap(self.image, mapsx, mapsy)
        MoilUtils.showImageToLabel(self.wind_detected_in, anypoint, 200)
        return anypoint

    def createAlphaBeta_out(self, x, y):
        """
        This function is useful for making alpha, beta in images on zone 2
        """
        alpha, beta = self.moildev_out.get_alpha_beta(x, y, mode=2)
        mapsx, mapsy = self.moildev_out.maps_anypoint(alpha, beta, zoom=20, mode=2)
        anypoint_out = MoilUtils.remap(self.image, mapsx, mapsy)
        MoilUtils.showImageToLabel(self.wind_detected_out, anypoint_out, 200)
        return anypoint_out

    @classmethod
    def CenterGravity(cls, maps_x, maps_y):
        """
        This function will be used to create the midpoint of map_X and maps_Y in the original image
        """
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
        """
        This function is used to crop plate images, when the program detects plates automatically and
        displayed as the result of the detection plate
        """
        frame2 = frame[box[0][1]:box[0][1] + box[0][3], box[0][0]:box[0][0] + box[0][2]]
        return frame2

    def show_image(self):
        """
        This function will be used to outline the polygons in the original image
        """
        image_draw = self.image.copy()
        image_draw = MoilUtils.drawPolygon(image_draw, self.maps_x_in, self.maps_y_in)
        image_draw = MoilUtils.drawPolygon(image_draw, self.maps_x_out, self.maps_y_out)
        # MoilUtils.showImageToLabel(self.original_fisheye, image_draw, 600)

    def save_image_inside(self):
        """
        This function will be used when saving Anypoint images in Zone 1
        """
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        cv2.imwrite("./images/anypoint_inside/img_in_" + curr_datetime + ".jpg", self.anypoint_in)
        print("save image inside")

    def save_image_outside(self):
        """
        This function will be used when saving Anypoint images in Zone 2
        """
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        cv2.imwrite("./images/anypoint_outside/img_out_" + curr_datetime + ".jpg", self.anypoint_out)
        print("save image outside")

    def valueChange_inside(self):
        """
        This function will be used when the user will configure the settings to get the results from
        picture on Zone 1 with direct control
        """
        alpha = self.val_alpha_in.value()
        beta = self.val_beta_in.value()
        zoom = self.val_zoom_in.value()

        if self.mode_1_in.isChecked():
            mode = 1
        elif self.mode_2_in.isChecked():
            mode = 2
        else:
            mode = 1

        self.maps_x_in, self.maps_y_in = self.moildev_in.maps_anypoint(alpha, beta, zoom, mode)
        self.x_in, self.y_in, self.center_fish, self.width, self.height = self.CenterGravity(self.maps_x_in,
                                                                                             self.maps_y_in)
        self.anypoint_zone_1()
        if self.maps_x_in is not None and self.maps_x_out is not None:
            self.show_image()

    def valueChange_outside(self):
        """
        This function will be used when the user will configure the settings to get the results from
        picture on Zone 2 with direct control
        """
        alpha = self.val_alpha_out.value()
        beta = self.val_beta_out.value()
        zoom = self.val_zoom_out.value()

        if self.mode_1_out.isChecked():
            mode = 1
        elif self.mode_2_out.isChecked():
            mode = 2
        else:
            mode = 1

        self.maps_x_out, self.maps_y_out = self.moildev_out.maps_anypoint(alpha, beta, zoom, mode)
        self.x_out, self.y_out, self.center_fish, self.width, self.height = self.CenterGravity(self.maps_x_out,
                                                                                               self.maps_y_out)

        self.anypoint_zone_2()
        if self.maps_x_in is not None and self.maps_x_out is not None:
            self.show_image()

    # Start from moilapps
    def mouse_event_image_source(self, e):
        """
        This function will be used when the user will click on the image to get the pixels to be seen and show on
        Zone 1 and Zone 2
        """
        print("click even normal fisheye")
        if e.button() == Qt.MouseButton.LeftButton:
            pos_x = round(e.position().x())
            pos_y = round(e.position().y())
            if self.image is None:
                print("no image")
            else:
                ratio_x, ratio_y = MoilUtils.ratio_image_to_label(self.original_fisheye, self.image)
                icx_front = round(pos_x * ratio_x)
                icy_front = round(pos_y * ratio_y)

                if self.radioButton_inside.isChecked():
                    print("inside")
                    if self.mode_1_in.isChecked():
                        mode = 1
                    elif self.mode_2_in.isChecked():
                        mode = 2
                    else:
                        mode = 1

                    alpha, beta = self.moildev_in.get_alpha_beta(icx_front, icy_front, mode)
                    self.blockSignals()
                    self.val_alpha_in.setValue(alpha)
                    self.val_beta_in.setValue(beta)
                    self.unblockSignals()
                    self.valueChange_inside()
                    self.anypoint_zone_1()
                    self.anypoint_zone_2()
                    self.show_image()
                if self.radioButton_outside.isChecked():
                    print("outside")
                    if self.mode_1_out.isChecked():
                        mode = 1
                    elif self.mode_2_out.isChecked():
                        mode = 2
                    else:
                        mode = 1

                    alpha, beta = self.moildev_out.get_alpha_beta(icx_front, icy_front, mode)
                    self.blockSignals()
                    self.val_alpha_out.setValue(alpha)
                    self.val_beta_out.setValue(beta)
                    self.unblockSignals()
                    self.valueChange_outside()
                    self.anypoint_zone_1()
                    self.anypoint_zone_2()
                    self.show_image()

        elif e.button() == Qt.MouseButton.RightButton:
            if self.image is not None:
                menu = QtWidgets.QMenu()
                save = menu.addAction("Open Image")
                info = menu.addAction("Show Info")
                save.triggered.connect(self.onclick_open_camera)
                # info.triggered.connect(self.onclick_help)
                menu.exec(e.globalPos())

    def mouse_event_image_anypoint(self, e):
        """
        This function will be used for perspective transform by doing 4-points on objects in Zone 1
        """
        print("click anypoint in")
        if e.button() == Qt.MouseButton.LeftButton:
            pos_x = round(e.position().x())
            pos_y = round(e.position().y())
            if self.image is None:
                print("no image")
            else:
                ratio_x, ratio_y = MoilUtils.ratio_image_to_label(self.wind_inside_image, self.anypoint_in_draw)
                icx_front = round(pos_x * ratio_x)
                icy_front = round(pos_y * ratio_y)
                coor = [icx_front, icy_front]

                if len(self.point_in) <= 4:
                    self.point_in.append(coor)

                for i, value in enumerate(self.point_in):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = self.point_in[i]
                    fontScale = 2
                    color = (255, 0, 0)
                    thickness = 5
                    self.anypoint_in_draw = cv2.putText(self.anypoint_in_draw, str(i + 1), (org[0], org[1]), font, fontScale,
                                                   color, thickness, cv2.LINE_AA)
                    cv2.circle(self.anypoint_in_draw, (org[0], org[1]), 15, (0, 0, 255), -1)

                if len(self.point_in) == 4:
                    self.image_click_plate = self.perspective_in(self.anypoint_in)
                    cv2.imwrite("image_in.jpg", self.image_click_plate)

                if len(self.point_in) > 4:
                    self.point_in = []
                    self.anypoint_in_draw = self.anypoint_in.copy()

                self.show_image_anypoint_draw()

    def mouse_event_image_anypoint_out(self, e):
        """
        This function will be used for perspective transform by doing 4-points on objects in Zone 2
        """
        print("click anypoint out")
        if e.button() == Qt.MouseButton.LeftButton:
            pos_x = round(e.position().x())
            pos_y = round(e.position().y())
            if self.image is None:
                print("no image")
            else:
                ratio_x, ratio_y = MoilUtils.ratio_image_to_label(self.wind_outsid_image, self.anypoint_out_draw)
                icx_front = round(pos_x * ratio_x)
                icy_front = round(pos_y * ratio_y)
                coor = [icx_front, icy_front]

                if len(self.point_out) <= 4:
                    self.point_out.append(coor)

                for i, value in enumerate(self.point_out):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = self.point_out[i]
                    fontScale = 2
                    color = (255, 0, 0)
                    thickness = 5
                    self.anypoint_out_draw = cv2.putText(self.anypoint_out_draw, str(i + 1), (org[0], org[1]), font,
                                                        fontScale,
                                                        color, thickness, cv2.LINE_AA)
                    cv2.circle(self.anypoint_out_draw, (org[0], org[1]), 15, (0, 0, 255), -1)

                if len(self.point_out) == 4:
                    self.image_click_plate_zone2 = self.perspective_out(self.anypoint_out)
                    cv2.imwrite("image_out.jpg", self.image_click_plate_zone2)

                if len(self.point_out) > 4:
                    self.point_out = []
                    self.anypoint_out_draw = self.anypoint_out.copy()

                self.show_image_anypoint_draw_out()
    # End from moilapps

    # Start aji guide source-code
    def perspective_in(self, image):
        """
        This function is to create a perspective transform in zone 1
        """
        pts1 = np.float32(self.point_in)
        pts2 = np.float32([[0, 0], [200, 0], [0, 100], [200, 100]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (200, 100))

    def perspective_out(self, image):
        """
        This function is to create a perspective transform in zone 2
        """
        pts1 = np.float32(self.point_out)
        pts2 = np.float32([[0, 0], [200, 0], [0, 100], [200, 100]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (200, 100))

    def rotate_value_ori(self, image):
        """
        This function will be used when rotating the image in Normal Fisheye
        """
        rotate = self.rotate_ori.value()
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotate, scale=1)
        return cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    def rotate_value_in(self, image):
        """
        This function will be used when rotating the image in Zone 1
        """
        rotate = self.rotate_in.value()
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotate, scale=1)
        return cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    def rotate_value_out(self, image):
        """
        This function will be used when rotating the image in Zone 2
        """
        rotate = self.rotate_out.value()
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotate, scale=1)
        return cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    def onclick_play_video(self):
        """
        This function will be used to play the video by clicking the play button on the user interface
        """
        if self.image is not None:
            if self.btn_play_pouse_3.isChecked():
                self.timer.start()
            else:
                self.timer.stop()

    def onclick_prev_video(self):
        """
        This function will be used to previous the video by clicking the previous button on the user interface
        """
        self.btn_prev_video_3.setChecked(False)
        self.rewind_video()
        self.timer.stop()

    def onclick_stop_video(self):
        """
        This function will be used to stop the video by clicking the stop button on the user interface
        """
        self.btn_play_pouse_3.setChecked(False)
        self.stop_video()
        self.timer.stop()

    def onclick_skip_video(self):
        """
        This function will be used to skip the video by clicking the skip button on the user interface
        """
        self.btn_skip_video_3.setChecked(False)
        self.forward_video()
        self.timer.stop()

    def rewind_video(self):
        """
        This function will be used to rewind the video by clicking the rewind button on the user interface
        """
        fps = self.video.get(cv2.CAP_PROP_FPS)
        position = self.data_properties.properties_video["pos_frame"] - 5 * fps
        self.video.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.next_frame()

    def stop_video(self):
        """
        This function will be used to stop the video by clicking the stop button on the user interface
        """
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.next_frame()

    def forward_video(self):
        """
        This function will be used to forward the video by clicking the forward button on the user interface
        """
        fps = self.video.get(cv2.CAP_PROP_FPS)
        position = self.data_properties.properties_video["pos_frame"] + 5 * fps
        self.video.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.next_frame()

    def onclick_slider_video(self, value):
        """
        This function will be used to clicked the slider on the user interface
        """
        value_max = self.slider_Video_3.maximum()
        self.slider_controller(value, value_max)

    def slider_controller(self, value, slider_maximum):
        """
        This function will be used to control the slider on the user interface
        """
        dst = self.data_properties.properties_video["frame_count"] * value / slider_maximum
        self.video.set(cv2.CAP_PROP_POS_FRAMES, dst)
        self.next_frame()

    def blockSignals(self):
        """
        This function will be used to grant True access to the setting configuration
        """
        self.rotate_ori.blockSignals(True)
        self.val_alpha_in.blockSignals(True)
        self.val_beta_in.blockSignals(True)
        self.val_zoom_in.blockSignals(True)
        self.rotate_in.blockSignals(True)
        self.val_alpha_out.blockSignals(True)
        self.val_beta_out.blockSignals(True)
        self.val_zoom_out.blockSignals(True)
        self.rotate_out.blockSignals(True)

    def unblockSignals(self):
        """
        This function will be used to grant False access to the setting configuration
        """
        self.rotate_ori.blockSignals(False)
        self.val_alpha_in.blockSignals(False)
        self.val_beta_in.blockSignals(False)
        self.val_zoom_in.blockSignals(False)
        self.rotate_in.blockSignals(False)
        self.val_alpha_out.blockSignals(False)
        self.val_beta_out.blockSignals(False)
        self.val_zoom_out.blockSignals(False)
        self.rotate_out.blockSignals(False)

