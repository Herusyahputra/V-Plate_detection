import json
from .resource_icon import *

try:
    from PyQt6 import QtWidgets, QtGui, QtCore
    pyqt_version = "pyqt6"

except:
    from PyQt5 import QtWidgets, QtGui, QtCore

    pyqt_version = "pyqt5"

type_camera = None


def set_icon_select_camera_type(comboBox_cam_type):
    comboBox_cam_type.setStyleSheet("QComboBox::drop-down {"
                                    "background-image: url(icons:chevron-down.svg);}")


def camera_type(camera_parameter, theme):
    """
    This function allows a user to choose what parameter will be used. this function will open a dialog,
    and you can select the parameter available from Combobox.

    return:
        cls.__type_camera : load camera type

    - Example:

    .. code-block:: python

        type_camera = MoilUtils.selectCameraType()
    """
    new_list = []

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    with open(camera_parameter) as f:
        data_parameter = json.load(f)
    for key in data_parameter.keys():
        new_list.append(key)
    Dialog = QtWidgets.QDialog()
    Dialog.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
    Dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
    Dialog.setObjectName("Dialog")
    Dialog.resize(298, 153)
    color = "rgb(220, 220, 220)"
    color_font = "rgb(20, 20, 20)"
    if theme == "dark":
        color = "#1C304A"
        color_font = "rgb(220, 220, 220)"

    Dialog.setStyleSheet("QFrame {\n"
                         " background-color:" + color + ";\n"
                                                        "color: " + color_font + ";\n"
                                                                                 "    border-radius: 20px;\n"
                                                                                 "    font: 12pt \"Segoe UI\";\n"
                                                                                 "}\n"
                                                                                 "\n"
                                                                                 "QComboBox{\n"
                                                                                 "    color: rgb(0, 0, 0);\n"
                                                                                 "    background-color: rgb(255, 255, 255);\n"
                                                                                 "    border-radius: 4px;\n"
                                                                                 "    border: 1px solid rgb(33, 37, 43);\n"
                                                                                 "    padding: 5px;\n"
                                                                                 "    padding-left: 10px;\n"
                                                                                 "}\n"
                                                                                 "QComboBox:hover{\n"
                                                                                 "    border: 2px solid rgb(64, 71, 88);\n"
                                                                                 "\n"
                                                                                 "}\n"
                                                                                 "\n"
                                                                                 "QComboBox::drop-down {\n"
                                                                                 "    subcontrol-origin: padding;\n"
                                                                                 "    subcontrol-position: top right;\n"
                                                                                 "    width: 25px; \n"
                                                                                 "    border-left-width: 3px;\n"
                                                                                 "    border-left-color: rgba(39, 44, 54, 150);\n"
                                                                                 "    border-left-style: solid;\n"
                                                                                 "    border-top-right-radius: 3px;\n"
                                                                                 "    border-bottom-right-radius: 3px;    \n"
                                                                                 "    background-position: center;\n"
                                                                                 "    background-repeat: no-reperat;\n"
                                                                                 " }\n"
                                                                                 "\n"
                                                                                 "QComboBox QAbstractItemView {\n"
                                                                                 "    color: rgb(0, 0, 0);    \n"
                                                                                 "    background-color: rgb(255, 255, 255);\n"
                                                                                 "    padding:5px;\n"
                                                                                 "    selection-background-color: rgb(39, 44, 54);\n"
                                                                                 "}")
    verticalLayout = QtWidgets.QVBoxLayout(Dialog)
    verticalLayout.setObjectName("verticalLayout")
    frame_main = QtWidgets.QFrame(Dialog)
    frame_main.setMinimumSize(QtCore.QSize(280, 0))
    frame_main.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
    frame_main.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
    frame_main.setObjectName("frame_main")
    verticalLayout_2 = QtWidgets.QVBoxLayout(frame_main)
    verticalLayout_2.setContentsMargins(20, 5, 20, 10)
    verticalLayout_2.setSpacing(5)
    verticalLayout_2.setObjectName("verticalLayout_2")
    label = QtWidgets.QLabel(frame_main)
    label.setMinimumSize(QtCore.QSize(0, 40))
    font = QtGui.QFont()
    font.setFamily("Segoe UI")
    font.setPointSize(12)
    font.setBold(False)
    font.setItalic(False)
    font.setWeight(50)
    label.setFont(font)
    label.setStatusTip("")
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setText("Select Camera Type !!!!")
    label.setObjectName("label")
    verticalLayout_2.addWidget(label)
    comboBox_cam_type = QtWidgets.QComboBox(frame_main)
    comboBox_cam_type.setMinimumSize(QtCore.QSize(0, 30))
    comboBox_cam_type.addItems(new_list)
    set_icon_select_camera_type(comboBox_cam_type)
    comboBox_cam_type.setObjectName("comboBox_cam_type")
    verticalLayout_2.addWidget(comboBox_cam_type)
    spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Policy.Minimum,
                                       QtWidgets.QSizePolicy.Policy.Fixed)
    verticalLayout_2.addItem(spacerItem)
    buttonBox = QtWidgets.QDialogButtonBox(frame_main)
    buttonBox.setStatusTip("")
    buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
    buttonBox.setStandardButtons(
        QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
    buttonBox.setObjectName("buttonBox")
    verticalLayout_2.addWidget(buttonBox)
    verticalLayout.addWidget(frame_main)
    buttonBox.accepted.connect(lambda: accept_btn(Dialog, comboBox_cam_type))
    buttonBox.rejected.connect(lambda: reject_btn(Dialog))
    Dialog.exec()
    return type_camera


def accept_btn(dialog, msg):
    """
    This function is to accept dialog, msg for camera type from the button which user clicked
    in the user interface

    Args:
        dialog: show dialog always in front of the user interface
        msg: accept msg current text from the camera used

    Returns:
        This function is None
    """
    global type_camera
    dialog.accept()
    type_camera = msg.currentText()


def reject_btn(dialog):
    """
    This function is to given reject on dialog if the camera type it's not in accordance
    Args:
        dialog: the rejection to show dialog in front of the user interface
    Returns:
        This function is None
    """
    global type_camera
    dialog.reject()
    type_camera = None
