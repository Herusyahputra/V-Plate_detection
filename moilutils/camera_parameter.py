import json
import sys
from .resource_icon import *

try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from .ui_camera_parameter_form_pyqt6 import Ui_Dialog
    pyqt_version = "pyqt6"

except:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from .ui_camera_parameter_form_pyqt5 import Ui_Dialog
    pyqt_version = "pyqt5"


class SetIconToFormParams:
    def __init__(self, main_ui):
        super(SetIconToFormParams, self).__init__()
        self.ui = main_ui
        self.adding_pixmap()
        self.set_icon_button()

    def adding_pixmap(self):
        self.ui.label.setPixmap(QtGui.QPixmap("icons:moil-icon.png"))

    def set_icon_button(self):
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icons:help-circle.svg"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.ui.btn_help.setIcon(icon1)

        icon1.addPixmap(QtGui.QPixmap("icons:light/cil-x.png"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.ui.btn_close.setIcon(icon1)

        icon1.addPixmap(QtGui.QPixmap("icons:light/cil-loop-circular.png"), QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off)
        self.ui.btn_synchronize.setIcon(icon1)


class CameraParametersForm(Ui_Dialog):
    def __init__(self, recent_win, camera_parameter_path):
        super(CameraParametersForm, self).__init__()
        self.recent_win = recent_win
        self.setupUi(self.recent_win)

        SetIconToFormParams(self)

        self.parameter_path = camera_parameter_path

        self.main_controller = None
        self.camera_name = None
        self.camera_fov = None
        self.sensor_width = None
        self.sensor_height = None
        self.image_height = None
        self.image_width = None
        self.icy = None
        self.icx = None
        self.ratio = None
        self.calibration_ratio = None
        self.parameter_0 = None
        self.parameter_1 = None
        self.parameter_2 = None
        self.parameter_3 = None
        self.parameter_4 = None
        self.parameter_5 = None

        self.centralwidget.setStyleSheet(style_appearance)
        self.recent_win.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.recent_win.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        with open(self.parameter_path) as f:
            self.data = json.load(f)
        self.update_list_camera()
        self.handle_combo_box()

        self.frame.mouseMoveEvent = self.moveWindow
        self.frame.mousePressEvent = self.mousePressEvent
        self.btn_close.clicked.connect(self.recent_win.close)
        self.list_camera.currentIndexChanged.connect(self.handle_combo_box)
        self.btn_open_file.clicked.connect(self.add_new_parameter_from_file)
        self.btn_save.clicked.connect(self.save_parameter_from_ui)
        self.btn_clean_all.clicked.connect(self.clean_parameter_in_ui)
        self.btn_delete.clicked.connect(self.delete_camera_parameter)
        self.btn_synchronize.clicked.connect(self.synchronize_data)
        self.btn_help.clicked.connect(self.open_help)

    def moveWindow(self, event):
        if pyqt_version == "pyqt6":
            if event.buttons() == QtCore.Qt.MouseButton.LeftButton:
                self.recent_win.move(self.recent_win.pos() + event.globalPosition().toPoint() - self.dragPos)
                self.dragPos = event.globalPosition().toPoint()
                event.accept()
        else:
            if event.buttons() == QtCore.Qt.LeftButton:
                delta = QtCore.QPoint(event.globalPos() - self.dragPos)
                self.recent_win.move(self.recent_win.x() + delta.x(), self.recent_win.y() + delta.y())
                self.dragPos = event.globalPos()
                event.accept()

    def mousePressEvent(self, event):
        if pyqt_version == "pyqt6":
            self.dragPos = event.globalPosition().toPoint()
        else:
            self.dragPos = event.globalPos()

    def update_list_camera(self):
        self.list_camera.blockSignals(True)
        self.list_camera.clear()
        parameter_list = []
        for key in self.data.keys():
            parameter_list.append(key)
        self.list_camera.addItems(parameter_list)
        self.list_camera.blockSignals(False)

    def handle_combo_box(self):
        if self.list_camera.count() != 0:
            camera_name = self.list_camera.currentText()
            key = list(self.data[camera_name])
            if camera_name in self.data.keys():
                self.camera_name = self.data[camera_name][key[0]]
                self.camera_fov = self.data[camera_name][key[1]]
                self.sensor_width = self.data[camera_name][key[2]]
                self.sensor_height = self.data[camera_name][key[3]]
                self.icx = self.data[camera_name][key[4]]
                self.icy = self.data[camera_name][key[5]]
                self.ratio = self.data[camera_name][key[6]]
                self.image_width = self.data[camera_name][key[7]]
                self.image_height = self.data[camera_name][key[8]]
                self.calibration_ratio = self.data[camera_name][key[9]]
                self.parameter_0 = self.data[camera_name][key[10]]
                self.parameter_1 = self.data[camera_name][key[11]]
                self.parameter_2 = self.data[camera_name][key[12]]
                self.parameter_3 = self.data[camera_name][key[13]]
                self.parameter_4 = self.data[camera_name][key[14]]
                self.parameter_5 = self.data[camera_name][key[15]]
                self.fill_the_properties_camera()

    def fill_the_properties_camera(self):
        self.label_camera_name.setText(self.camera_name)
        self.label_camera_fov.setText(str(self.camera_fov))
        self.label_cam_sensor_width.setText(str(self.sensor_width))
        self.label_cam_sensor_height.setText(str(self.sensor_height))
        self.label_image_center_X.setText(str(self.icx))
        self.label_image_center_Y.setText(str(self.icy))
        self.label_ratio.setText(str(self.ratio))
        self.label_image_width.setText(str(self.image_width))
        self.label_image_height.setText(str(self.image_height))
        self.label_calib_ratio.setText(str(self.calibration_ratio))
        self.label_parameter0.setText(str(self.parameter_0))
        self.label_parameter1.setText(str(self.parameter_1))
        self.label_parameter2.setText(str(self.parameter_2))
        self.label_parameter3.setText(str(self.parameter_3))
        self.label_parameter4.setText(str(self.parameter_4))
        self.label_parameter5.setText(str(self.parameter_5))

    def add_new_parameter_from_file(self):
        parameter_path = self.select_file(None, "Select Parameter !", "../", "Parameter Files (*.json)")
        if parameter_path:
            with open(parameter_path) as f:
                camera = json.load(f)
            self.label_camera_name.setText(camera["cameraName"])
            self.label_camera_fov.setText(str(""))
            self.label_cam_sensor_width.setText(str(camera["cameraSensorWidth"]))
            self.label_cam_sensor_height.setText(str(camera["cameraSensorHeight"]))
            self.label_image_center_X.setText(str(camera["iCx"]))
            self.label_image_center_Y.setText(str(camera["iCy"]))
            self.label_ratio.setText(str(camera["ratio"]))
            self.label_image_width.setText(str(camera["imageWidth"]))
            self.label_image_height.setText(str(camera["imageHeight"]))
            self.label_calib_ratio.setText(str(camera["calibrationRatio"]))
            self.label_parameter0.setText(str(camera["parameter0"]))
            self.label_parameter1.setText(str(camera["parameter1"]))
            self.label_parameter2.setText(str(camera["parameter2"]))
            self.label_parameter3.setText(str(camera["parameter3"]))
            self.label_parameter4.setText(str(camera["parameter4"]))
            self.label_parameter5.setText(str(camera["parameter5"]))
            camera_name = str(camera["cameraName"])
            self.data[camera_name] = {}
            self.update_list_camera()
            self.list_camera.blockSignals(True)
            self.list_camera.setCurrentText(camera_name)
            self.list_camera.blockSignals(False)
            self.save_parameter_from_ui()

    def save_parameter_from_ui(self):
        camera_name = self.list_camera.currentText()
        if self.label_camera_name.text() != "":
            if self.label_camera_name.text() != camera_name:
                camera_name = self.label_camera_name.text()
                self.data[self.label_camera_name.text()] = {}
            self.data[camera_name]["cameraName"] = self.label_camera_name.text()
            self.data[camera_name]["cameraFov"] = int(self.label_camera_fov.text()) \
                if self.label_camera_fov.text() != "" else 220
            self.data[camera_name]["cameraSensorWidth"] = float(self.label_cam_sensor_width.text())
            self.data[camera_name]["cameraSensorHeight"] = float(self.label_cam_sensor_height.text())
            self.data[camera_name]["iCx"] = int(self.label_image_center_X.text())
            self.data[camera_name]["iCy"] = int(self.label_image_center_Y.text())
            self.data[camera_name]["ratio"] = float(self.label_ratio.text())
            self.data[camera_name]["imageWidth"] = int(self.label_image_width.text())
            self.data[camera_name]["imageHeight"] = int(self.label_image_height.text())
            self.data[camera_name]["calibrationRatio"] = float(self.label_calib_ratio.text())
            self.data[camera_name]["parameter0"] = float(self.label_parameter0.text())
            self.data[camera_name]["parameter1"] = float(self.label_parameter1.text())
            self.data[camera_name]["parameter2"] = float(self.label_parameter2.text())
            self.data[camera_name]["parameter3"] = float(self.label_parameter3.text())
            self.data[camera_name]["parameter4"] = float(self.label_parameter4.text())
            self.data[camera_name]["parameter5"] = float(self.label_parameter5.text())
            self.update_list_camera()
            self.list_camera.setCurrentText(camera_name)
            with open(self.parameter_path, 'w') as f:
                json.dump(self.data, f)

    def delete_camera_parameter(self):
        camera_name = self.list_camera.currentText()
        self.data.pop(camera_name)
        self.clean_parameter_in_ui()
        with open(self.parameter_path, 'w') as f:
            json.dump(self.data, f)
        self.update_list_camera()

    def synchronize_data(self):
        QtWidgets.QMessageBox.information(None, "Info!!", "Under developing !!")

    def open_help(self):
        QtWidgets.QMessageBox.information(None, "Information !!", "Camera Parameter form, you can add from files, \n"
                                                                  "or can you can typing the parameter information \n"
                                                                  "the form !!! ")

    def clean_parameter_in_ui(self):
        self.label_camera_name.setText("")
        self.label_camera_fov.setText("")
        self.label_cam_sensor_width.setText("")
        self.label_cam_sensor_height.setText("")
        self.label_image_center_X.setText("")
        self.label_image_center_Y.setText("")
        self.label_ratio.setText("")
        self.label_image_width.setText("")
        self.label_image_height.setText("")
        self.label_calib_ratio.setText("")
        self.label_parameter0.setText("")
        self.label_parameter1.setText("")
        self.label_parameter2.setText("")
        self.label_parameter3.setText("")
        self.label_parameter4.setText("")
        self.label_parameter5.setText("")

    @classmethod
    def select_file(cls, parent=None, title="Open file", dir_path="/camera_parameter/", file_filter=""):
        """
        Find the file path from the directory computer.

        Args:
            parent (): The parent windows to show dialog always in front of user interface
            title: the title window of open dialog
            file_filter: determine the specific file want to search
            dir_path: Navigate to specific directory

        return:
            file_path: location
        """
        if pyqt_version == "pyqt6":
            option = QtWidgets.QFileDialog.Option.DontUseNativeDialog
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, dir_path,
                                                                 file_filter, options=option)
        else:
            options = QtWidgets.QFileDialog.DontUseNativeDialog
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, dir_path,
                                                                 file_filter,
                                                                 options=options)
        return file_path

style_appearance = """
    QWidget {
        color: rgb(221, 221, 221);
        font: 10pt "Segoe UI";
        border: none;
    }

    #frame_main{
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
        border: 2px solid rgb(52, 59, 72);
        padding: 5px;
        padding-left: 10px;
    }
    QComboBox:hover{
        border: 2px solid rgb(82, 94, 88);
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
    }
"""
