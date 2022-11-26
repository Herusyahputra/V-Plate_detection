import cv2
from .resource_icon import *


try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from .ui_camera_source_pyqt6 import Ui_Dialog
    pyqt_version = "pyqt6"

except:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from .ui_camera_source_pyqt5 import Ui_Dialog
    pyqt_version = "pyqt5"


def check_Port_USB_Camera():
    """
    Detect the USB camera port available and show it on message box prompt.

    Returns:

    """
    all_camera_idx_available = []
    for camera_idx in range(5):
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            all_camera_idx_available.append(camera_idx)
            cap.release()

    msgbox = QtWidgets.QMessageBox()
    msgbox.setWindowTitle("Camera Port Available")
    msgbox.setText(
        "Select the port camera from the number in list !! \n"
        "Available Port = " + str(all_camera_idx_available))
    msgbox.exec()


class CameraSource(Ui_Dialog):
    def __init__(self, RecentWindow, theme="light"):
        """
        Create class controllers open camera with inheritance from Ui Dialog Class.

        Args:
            RecentWindow ():
        """
        super(CameraSource, self).__init__()
        self.recent_win = RecentWindow
        self.setupUi(self.recent_win)

        self.central_widget.setStyleSheet(style_appearance)
        self.recent_win.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.recent_win.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        self.camera_source = None
        self.camera_stream_link.setText('http://<Type your Ip address here>:8000/stream.mjpg')
        self.handle_activated_comboBox()
        self.comboBox_camera_sources.activated.connect(self.handle_activated_comboBox)
        self.btn_detect_port_Camera.clicked.connect(check_Port_USB_Camera)
        self.buttonBox.accepted.connect(self.onclick_comboBox_oke)
        self.buttonBox.rejected.connect(self.onclick_comboBox_cancel)

    def handle_activated_comboBox(self):
        """
        Handle the selection from comboBox of source camera.

        Returns:

        """
        if self.comboBox_camera_sources.currentText() == "USB Camera":
            self.portCamera.show()
            self.btn_detect_port_Camera.show()
            self.camera_stream_link.hide()
            self.label_3.setText("Select Port :")

        else:
            self.label_3.setText("Camera Link :")
            self.camera_stream_link.show()
            self.portCamera.hide()
            self.btn_detect_port_Camera.hide()

    def camera_source_used(self):
        """
        This function will return the source of camera used depend on what the camera use.

        Returns:
            camera source
        """
        if self.comboBox_camera_sources.currentText() == "USB Camera":
            self.camera_source = int(self.portCamera.currentText())
        else:
            self.camera_source = self.camera_stream_link.text()

    def onclick_comboBox_oke(self):
        """
        Open the camera following the parent function and close the dialog window.

        Returns:

        """
        self.camera_source_used()
        self.recent_win.close()

    def onclick_comboBox_cancel(self):
        """
        close the window when you click the buttonBox cancel.

        Returns:

        """
        self.recent_win.close()

# class SelectCameraSources(QtWidgets.QDialog):
#     def __init__(self, view_controller):
#         super(SelectCameraSources, self).__init__()
#         self.ui = Ui_Dialog()
#         self.ui.setupUi(self)
#
#         self.view_controller = view_controller
#
#         self.ui.central_widget.setStyleSheet(style_appearance)
#         self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
#         self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
#
#         self.camera_path = None
#         self.option_usb_or_cam()
#         self.connect_button()
#
#     def connect_button(self):
#         self.ui.comboBox_camera_sources.currentIndexChanged.connect(self.option_usb_or_cam)
#         self.ui.btn_detect_port_Camera.clicked.connect(self.check_port_usb_camera)
#         self.ui.buttonBox.accepted.connect(self.accepted)
#         self.ui.buttonBox.rejected.connect(self.rejected)
#
#     def accepted(self):
#         self.camera_source_used()
#         self.close()
#
#     def camera_source_used(self):
#         """
#         This function will return the source of camera used depend on what the camera use.
#
#         Returns:
#             camera source
#         """
#         if self.ui.comboBox_camera_sources.currentText() == "USB Camera":
#             self.view_controller.model.media_path = int(self.ui.portCamera.currentText())
#         else:
#             self.view_controller.model.media_path = self.ui.camera_stream_link.text()
#         # camera_type = select_camera_type(self.view_controller.main_controller.camera_params.data_parameter,
#         #                                  self.view_controller.THEME)
#         # if camera_type:
#         #     self.view_controller.main_controller.set_moildev_object(camera_type)
#         #     self.view_controller.main_controller.running_video()
#         #     self.view_controller.show_to_ui.update_image_to_ui()
#
#     def option_usb_or_cam(self):
#         if self.ui.comboBox_camera_sources.currentText() == "USB Camera":
#             self.ui.portCamera.show()
#             self.ui.btn_detect_port_Camera.show()
#             self.ui.camera_stream_link.hide()
#             self.ui.label_3.setText("Select Port :")
#         if self.ui.comboBox_camera_sources.currentText() == "WEB Camera":
#             self.ui.label_3.setText("Camera Link :")
#             self.ui.camera_stream_link.show()
#             self.ui.portCamera.hide()
#             self.ui.btn_detect_port_Camera.hide()
#
#         else:
#             pass
#
#     @classmethod
#     def check_port_usb_camera(cls):
#         list_port_available = []
#         for camera_idx in range(5):
#             cap = cv2.VideoCapture(camera_idx)
#             if cap.isOpened():
#                 list_port_available.append(camera_idx)
#                 cap.release()
#         msgbox = QtWidgets.QMessageBox()
#         msgbox.setStyleSheet("color:white;background-color: rgb(37, 41, 48)")
#         msgbox.setWindowTitle("Camera Port Available")
#         # self.ui.central_widget.setStyleSheet(style_appearance)
#         # msgbox.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
#         # msgbox.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
#         msgbox.setText(
#             "Select the port camera from the number in list !! \n"
#             "Available Port = " + str(list_port_available))
#         msgbox.exec()
#
#     @classmethod
#     def accept_button(cls, dialog, msg):
#         global camera_path
#         dialog.accept()
#         # camera_path = msg.currentText()
#
#     @classmethod
#     def reject_btn(cls, dialog):
#         global camera_path
#         dialog.reject()
#         camera_type = None


style_appearance = """
    QWidget {
        color: rgb(221, 221, 221);
        font: 10pt "Segoe UI";
        border: none;
    }

    #frame{
        background-color: rgb(37, 41, 48);
        border: 1px solid rgb(44, 49, 58);
        border-radius: 10px;
    }

    #label_title{
        color: rgb(238,238,238);
        font: 14pt "Segoe UI";
    }

    QLineEdit {
        font: 9pt "Segoe UI";
        background-color: rgb(33, 37, 43);
        border-radius: 5px;
        border: 2px solid rgb(33, 37, 43);
        padding-left: 10px;
        selection-color: rgb(255, 255, 255);
        selection-background-color: rgb(255, 121, 198);
    }

    QLineEdit:hover {
        border: 2px solid rgb(64, 71, 88);
    }

    QLineEdit:focus {
        border: 2px solid rgb(91, 101, 124);
    }

    QComboBox{
        background-color: rgb(27, 29, 35);
        border-radius: 5px;
        border: 2px solid rgb(33, 37, 43);
        padding: 1px;
        padding-left: 15px;
    }
    #portCamera::drop-down{
        border:0Px;
    }

    QComboBox:hover{
        border: 2px solid rgb(64, 71, 88);
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 25px; 
        border-left-width: 3px;
        border-left-color: rgba(39, 44, 54, 150);
        border-left-style: solid;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
        background-image: url(icons:light/cil-arrow-bottom.png);	
        background-position: center;
        background-repeat: no-reperat;
     }
    QComboBox QAbstractItemView {
        color: rgb(255, 121, 198);	
        background-color: rgb(33, 37, 43);
        padding: 10px;
        selection-background-color: rgb(39, 44, 54);
    }

    QPushButton {
        border: 2px solid rgb(52, 59, 72);
        border-radius: 5px;	
        background-color: rgb(52, 59, 72);
    }

    QPushButton:hover {
        background-color: rgb(57, 65, 80);
        border: 2px solid rgb(61, 70, 86);
    }

    QPushButton:pressed {	
        background-color: rgb(35, 40, 49);
        border: 2px solid rgb(43, 50, 61);
    }"""
