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

        # Setting parameters for create anypoint maps inside
        self.val_alpha_out.valueChanged.connect(self.valueChange_outside)
        self.val_beta_out.valueChanged.connect(self.valueChange_outside)
        self.val_zoom_out.valueChanged.connect(self.valueChange_outside)

    def onclick_open_file(self):
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


    def get_value_slider_video(self, value):
        current_position = self.data_properties.properties_video["pos_frame"] * (value + 1) / \
                           self.data_properties.properties_video["frame_count"]
        return current_position

    def onclick_open_camera(self):
        print("test streaming")
        camera_link = "http://10.42.0.170:8000/stream.mjpg"
        self.video = cv2.VideoCapture(camera_link)

        self.moildev_in = MoilUtils.connectToMoildev("entaniya")
        self.moildev_out = MoilUtils.connectToMoildev("entaniya")

        self.valueChange_inside()
        self.valueChange_outside()
        self.next_frame()

    def next_frame_streaming(self):
        if self.video:
            success, self.image = self.video.read()
            if success:
                self.data_properties.properties_video["video"] = True
                self.timer.start()
                self.normal_fisheye()
                self.anypoint_zone_1()
                self.anypoint_zone_2()
                self.show_image()

    def save_to_record(self):
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
        print("normal image")
        self.ori_fisheye = self.image
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

            self.createAlphaBeta(int(self.x_in + real_distance_w), int(self.y_in + real_distance_h))
            MoilUtils.showImageToLabel(self.repository, frame_ori, 300)

        else:
            self.repository.setText("No detected")

        MoilUtils.showImageToLabel(self.original_fisheye, self.anypoint_ori_draw, 600)

    def anypoint_zone_1(self):
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
        # if self.position_in is not None:
        # MoilUtils.showImageToLabel(self.original_fisheye, self.anypoint_in_draw, 600)
        MoilUtils.showImageToLabel(self.wind_inside_image, self.anypoint_in_draw, 600)
        if self.image_click_plate is not None:
            MoilUtils.showImageToLabel(self.wind_detected_in_m, self.image_click_plate, 200)

    def anypoint_zone_2(self):
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

    def show_image_anypoint_draw_out(self):
        MoilUtils.showImageToLabel(self.wind_outsid_image, self.anypoint_out_draw, 600)
        if self.image_click_plate_zone2 is not None:
            MoilUtils.showImageToLabel(self.wind_detected_out_m, self.image_click_plate_zone2, 200)

    def createAlphaBeta(self, x, y):
        alpha, beta, = self.moildev_in.get_alpha_beta(x, y, mode=2)
        mapsx, mapsy = self.moildev_in.maps_anypoint(alpha, beta, zoom=20, mode=2)
        anypoint = MoilUtils.remap(self.image, mapsx, mapsy)
        MoilUtils.showImageToLabel(self.wind_detected_in, anypoint, 200)
        return anypoint

    def createAlphaBeta_out(self, x, y):
        alpha, beta = self.moildev_out.get_alpha_beta(x, y, mode=2)
        mapsx, mapsy = self.moildev_out.maps_anypoint(alpha, beta, zoom=20, mode=2)
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
        frame2 = frame[box[0][1]:box[0][1] + box[0][3], box[0][0]:box[0][0] + box[0][2]]
        return frame2

    def show_image(self):
        image_draw = self.image.copy()
        image_draw = MoilUtils.drawPolygon(image_draw, self.maps_x_in, self.maps_y_in)
        image_draw = MoilUtils.drawPolygon(image_draw, self.maps_x_out, self.maps_y_out)
        MoilUtils.showImageToLabel(self.original_fisheye, image_draw, 600)

    def save_image_inside(self):
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        cv2.imwrite("./images/anypoint_inside/img_in_" + curr_datetime + ".jpg", self.anypoint_in)
        print("save image inside")

    def save_image_outside(self):
        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        cv2.imwrite("./images/anypoint_outside/img_out_" + curr_datetime + ".jpg", self.anypoint_out)
        print("save image outside")

    def valueChange_inside(self):
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

    def mouse_event_image_source(self, e):
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

    # start aji guide source-code
    def perspective_in(self, image):
        # define four points on input image
        pts1 = np.float32(self.point_in)
        print("ok, test click perspective")

        # define the corresponding four points on output image
        pts2 = np.float32([[0, 0], [200, 0], [0, 100], [200, 100]])

        # get the perspective transform matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # transform the image using perspective transform matrix
        return cv2.warpPerspective(image, M, (200, 100))

    def perspective_out(self, image):
        # define four points on input image
        pts1 = np.float32(self.point_out)
        print("ok, test click perspective")

        # define the corresponding four points on output image
        pts2 = np.float32([[0, 0], [200, 0], [0, 100], [200, 100]])

        # get the perspective transform matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # transform the image using perspective transform matrix
        return cv2.warpPerspective(image, M, (200, 100))

    # start aji guide source-code
    def rotate_value_in(self, image):
        rotate = self.rotate_in.value()
        height, width = image.shape[:2]
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width / 2, height / 2)

        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotate, scale=1)

        # rotate the image using cv2.warpAffine
        return cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    def rotate_value_out(self, image):
        rotate = self.rotate_out.value()
        height, width = image.shape[:2]
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width / 2, height / 2)

        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotate, scale=1)

        # rotate the image using cv2.warpAffine
        return cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

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

    def stop_video(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.next_frame()

    def forward_video(self):
        fps = self.video.get(cv2.CAP_PROP_FPS)
        position = self.data_properties.properties_video["pos_frame"] + 5 * fps
        self.video.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.next_frame()

    def onclick_slider_video(self, value):
        value_max = self.slider_Video_3.maximum()
        self.slider_controller(value, value_max)

    def slider_controller(self, value, slider_maximum):
        dst = self.data_properties.properties_video["frame_count"] * value / slider_maximum
        self.video.set(cv2.CAP_PROP_POS_FRAMES, dst)
        self.next_frame()

    def blockSignals(self):
        self.val_alpha_in.blockSignals(True)
        self.val_beta_in.blockSignals(True)
        self.val_zoom_in.blockSignals(True)
        self.rotate_in.blockSignals(True)
        self.val_alpha_out.blockSignals(True)
        self.val_beta_out.blockSignals(True)
        self.val_zoom_out.blockSignals(True)
        self.rotate_out.blockSignals(True)

    def unblockSignals(self):
        self.val_alpha_in.blockSignals(False)
        self.val_beta_in.blockSignals(False)
        self.val_zoom_in.blockSignals(False)
        self.rotate_in.blockSignals(False)
        self.val_alpha_out.blockSignals(False)
        self.val_beta_out.blockSignals(False)
        self.val_zoom_out.blockSignals(False)
        self.rotate_out.blockSignals(False)
