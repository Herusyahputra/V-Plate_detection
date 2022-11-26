import datetime
import os
import shutil
import cv2
import numpy as np
from .Moildev import Moildev
import json
from .camera_source import CameraSource
from .camera_parameter import CameraParametersForm
from .camera_type import camera_type

try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    pyqt_version = "pyqt6"

except:
    from PyQt5 import QtWidgets, QtCore, QtGui
    pyqt_version = "pyqt5"


database_camera_parameters = "camera_parameters.json"


def select_type_camera(theme="dark"):
    """
    Generate a dialog for camera selection and return the name of the camera selected.

    Args:
        theme:
            dark mode
    Returns:
        camera name

    ..code-block: python

        select_type = mutils.select_type_camera()

    """
    camera_name = camera_type(database_camera_parameters, theme)
    return camera_name


def select_source_camera():
    """
    This function is for select source camera which one the detected on usb or http (browsing)

    Returns:
        source camera
    """
    open_cam_source = QtWidgets.QDialog()
    source_cam = CameraSource(open_cam_source)
    open_cam_source.exec()
    return source_cam.camera_source


def form_camera_parameter():
    """
    Generate a dialog for camera parameters setting.

    - Example:

    .. code-block:: python

        params = mutils.form_camera_parameter()

    """
    open_cam_params = QtWidgets.QDialog()
    CameraParametersForm(open_cam_params, database_camera_parameters)
    open_cam_params.exec()



def show_image_to_label(label, image, width, angle=0, plusIcon=False, scale_content=False):
    """
    Display an image to the label widget on the user interface.

    Args:
        scale_content:
        label: The label will contain image to show in your user interface
        image: Image that want to show on user interface
        width: the width of result image, this value will calculate the height following the ratio.
        angle: the angle of image
        plusIcon: Drawing the plus icons at the center of the image, by default this will be False.
                    Set it to True to show the icon.

    Returns:
        Showing image on the label

    - Example:

    .. code-block:: python

        image = MoilUtils.showImageToLabel(label, image, 400, 0, False)
    """
    print(scale_content)
    if scale_content is True:
        label.setScaledContents(True)

    else:
        label.setScaledContents(False)
        height = calculate_height(image, width)
        image = resize_image(image, width)
        label.setMinimumSize(QtCore.QSize(width, height))
        label.setMaximumSize(QtCore.QSize(width, height))

    image = rotate_image(image, angle)
    if plusIcon:
        # draw plus icons on image and show to label
        h, w = image.shape[:2]
        w1 = round((w / 2) - 10)
        h1 = round(h / 2)
        w2 = round((w / 2) + 10)
        h2 = round(h / 2)
        w3 = round(w / 2)
        h3 = round((h / 2) - 10)
        w4 = round(w / 2)
        h4 = round((h / 2)) + 10
        draw_line(image, w1, h1, w2, h2)
        draw_line(image, w3, h3, w4, h4)

    image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                         QtGui.QImage.Format.Format_RGB888).rgbSwapped()
    label.setPixmap(QtGui.QPixmap.fromImage(image))


def connect_to_moildev(type_camera, camera_parameter=None, parent=None):
    """
    Return a Moildev instance for a selected camera.

    Args:
        camera_parameter: choose camera used
        type_camera: load camera type
        parent: None

    Returns:
        Moildev
    """
    if camera_parameter is None:
        try:
            moildev = Moildev.Moildev(database_camera_parameters, type_camera)
        except:
            QtWidgets.QMessageBox.warning(
                parent,
                "Warning!!",
                "The image not support for this application, \n\nPlease contact developer!!")
            print("the image not support for this application, please contact developer!!")
            moildev = None

    else:
        try:
            moildev = Moildev.Moildev(camera_parameter, type_camera)
        except:
            QtWidgets.QMessageBox.warning(
                parent,
                "Warning!!",
                "The image not support for this application, \n\nPlease contact developer!!")
            print("the image not support for this application, please contact developer!!")
            moildev = None

    return moildev


def check_usb_camera_available():
    """
    Detect the USB camera port available and show it on message box prompt.

    Returns:
            a list of indices of available cameras
    """
    all_camera_idx_available = []
    for camera_idx in range(5):
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            all_camera_idx_available.append(camera_idx)
            cap.release()

    return all_camera_idx_available


def read_image(image_path):
    """
    Load an image from the specified file path (use the cv2.imread function to complete the task).
    If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format)
    then this method raises an error.

    Args:
        image_path : The path of image file

    return:
        Image: loaded image in numpy array format

    - Example:

    .. code-block:: python

        image = read_image(image_path)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("`{}` cannot be loaded".format(image_path))
    return image


def remap_image(image, map_x, map_y):
    """
    Take an image and a pair of X-Y maps generated by a Moildev instance as inputs,
     then return a remapped image using the maps.

    Args:
        image: Input image
        map_x: The mapping function in the x direction.
        map_y: The mapping function in the y direction.

    Returns:
        image: A remapped image, typically it would be an anypoint image or a panorama image.

    - Example:

    .. code-block:: python

        image_anypoint = remap_image(image, mapX_anypoint, mapY_anypoint)
    """
    image = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)
    return image


def select_file(parent=None, title="Open file", dir_path=".", file_filter=""):
    """
    Generate a dialog for file selection and return the path of the
    file selected. If no file is selected, an empty string is returned.

    Args:
        parent: The parent windows to show dialog always in front of user interface
        title: the title window of open dialog
        file_filter: determine the specific file want to search
        dir_path: Navigate to specific directory

    return:
        The path of the file selected.
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


def select_directory(parent=None, title='Select Folder'):
    """
    Generate a dialog for directory selection and return the path of the directory
    selected. If no directory is selected, an empty string is returned.


    Returns:
        The path of the directory selected


    """
    if pyqt_version == "pyqt6":
        option = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        directory = QtWidgets.QFileDialog.getExistingDirectory(parent, title, options=option)
    else:
        option = QtWidgets.QFileDialog.DontUseNativeDialog
        directory = QtWidgets.QFileDialog.getExistingDirectory(parent, title, options=option)
    return directory


def copy_directory(src_directory, dst_directory):
    """
    Recursively copy a directory tree to the destination directory.

    Args:
        src_directory: source path or original path folder
        dst_directory: destination directory.

    Returns:
        None
    """
    directoryName = os.path.basename(src_directory)
    destinationPath = os.path.join(dst_directory, directoryName)
    shutil.copytree(src_directory, destinationPath)


def resize_image(image, width):
    """
    Resize an image to one with a width given while maintaining its aspect ratio and
    return it.

    Args:
        image: original image
        width: desired image width

    Returns:
        A resized image

    """
    h, w = image.shape[:2]
    r = width / float(w)
    hi = round(h * r)
    result = cv2.resize(image, (width, hi),
                        interpolation=cv2.INTER_AREA)
    return result


def rotate_image(src, angle, center=None, scale=1.0):
    """
    Rotation of images are among the most basic operations under the broader class of
    Affine transformations. This function will return the image after turning clockwise
    or anticlockwise depending on the angle given.

    Args:
        src: original image
        angle: the value angle for turn the image
        center: determine the specific coordinate to rotate image
        scale: scale image

    Returns:
        dst image: rotated image

    - Example:

    .. code-block:: python

        image = rotate_image(image, 90)
    """
    h, w = src.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(src, m, (w, h))
    return rotated


def calculate_height(image, width):
    """
    Return the height value of an image by providing the width value. This high
    value is calculated by keeping the aspect ratio of the image.

    Args:
        image: original image
        width: size image we want

    Returns:
        height: height image

    - Example:

    .. code-block:: python

        height = calculate_height(image, 140)
    """

    h, w = image.shape[:2]
    r = width / float(w)
    height = round(h * r)
    return height


def draw_polygon(image, mapX, mapY):
    """
    Return image with a drawn polygon on it from mapX and mapY generated by maps anypoint or panorama.

    Args:
        image: Original image
        mapX: map image X from anypoint process
        mapY: map image Y from anypoint process

    return:
        image: map x, map y

    - Example:

    .. code-block:: python

        image = draw_polygon(image,mapX,mapY)
    """
    hi, wi = image.shape[:2]
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    X3 = []
    Y3 = []
    X4 = []
    Y4 = []

    x = 0
    while x < wi:
        a = mapX[0,]
        b = mapY[0,]
        ee = mapX[-1,]
        f = mapY[-1,]

        if a[x] == 0. or b[x] == 0.:
            pass
        else:
            X1.append(a[x])
            Y1.append(b[x])

        if f[x] == 0. or ee[x] == 0.:
            pass
        else:
            Y3.append(f[x])
            X3.append(ee[x])
        x += 10

    y = 0
    while y < hi:
        c = mapX[:, 0]
        d = mapY[:, 0]
        g = mapX[:, -1]
        h = mapY[:, -1]

        # eliminate the value 0 for map X
        if d[y] == 0. or c[y] == 0.:  # or d[y] and c[y] == 0.0:
            pass
        else:
            Y2.append(d[y])
            X2.append(c[y])

        # eliminate the value 0 for map Y
        if h[y] == 0. or g[y] == 0.:
            pass
        else:
            Y4.append(h[y])
            X4.append(g[y])

        # render every 10 times, it will be like 1, 11, 21 and so on.
        y += 10

    p = np.array([X1, Y1])
    q = np.array([X2, Y2])
    r = np.array([X3, Y3])
    s = np.array([X4, Y4])
    points = p.T.reshape((-1, 1, 2))
    points2 = q.T.reshape((-1, 1, 2))
    points3 = r.T.reshape((-1, 1, 2))
    points4 = s.T.reshape((-1, 1, 2))

    # Draw polyline on original image
    cv2.polylines(image, np.int32([points]), False, (0, 0, 255), 10)
    cv2.polylines(image, np.int32([points2]), False, (255, 0, 0), 10)
    cv2.polylines(image, np.int32([points3]), False, (0, 255, 0), 10)
    cv2.polylines(image, np.int32([points4]), False, (0, 255, 0), 10)
    return image


def write_camera_type(image_file, typeCamera):
    """
    This function for Read the camera used from metadata image.

    Args:
        image_file: image file path
        typeCamera: the name of type camera (string)

    Returns:
        None
    """
    img = pyexiv2.Image(image_file)
    pyexiv2.registerNs('a namespace for image', 'Image')
    img.modify_xmp({'Xmp.Image.cameraName': typeCamera})
    img.close()


def read_camera_type(image_file):
    """
    Read the camera used from metadata image using pyexiv2 library.

    Args:
        image_file: load file image

    Returns:
        camera type

    """
    img = pyexiv2.Image(image_file)
    try:
        camera_type = img.read_xmp()['Xmp.Image.cameraName']

    except:
        camera_type = None
    img.close()
    return camera_type


def draw_point(image, coordinate_point, radius=5):
    """
    This fucntion is for Drawing point on the image.

    Args:
        image: source image
        coordinate_point (): the coordinate point (x, y)
        radius: the size of the point (scale by radius)

    Returns:
        image
    """

    if coordinate_point is not None:
        w, h = image.shape[:2]
        if h >= 1000:
            cv2.circle(image, coordinate_point, radius, (200, 5, 200), 30, -1)
        else:
            cv2.circle(image, coordinate_point, radius, (200, 5, 200), -1)
    return image


def save_image(image, dst_directory, type_camera=None):
    """
    This function is for saved image on application

    Args:
        image: Source image want to save.
        dst_directory: destination directory
        type_camera: Type camera (string)

    Returns:
        the file name (string)
    """
    ss = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    name = dst_directory + "/" + str(ss) + ".png"
    cv2.imwrite(name, image)
    if type_camera is not None:
        write_camera_type(name, type_camera)
    return ss


def draw_line(image, coordinatePoint_1=None, coordinatePoint_2=None):
    """
    Draw line on the image from the coordinate given.

    Args:
        image: source image
        coordinatePoint_1: coordinate point 1 (x, y)
        coordinatePoint_2: coordinate point 2 (x, y)

    Returns:
        image with line drawn

    """
    # draw anypoint line
    if coordinatePoint_1 is None:
        h, w = image.shape[:2]
        if h >= 1000:
            cv2.line(image, (0, 0), (0, h), (255, 0, 0), 10)
            cv2.line(image, (0, 0), (w, 0), (0, 0, 255), 10)
            cv2.line(image, (0, h), (w, h), (0, 255, 0), 10)
            cv2.line(image, (w, 0), (w, h), (0, 255, 0), 10)
        else:
            cv2.line(image, (0, 0), (0, h), (255, 0, 0), 2)
            cv2.line(image, (0, 0), (w, 0), (0, 0, 255), 2)
            cv2.line(image, (0, h), (w, h), (0, 255, 0), 2)
            cv2.line(image, (w, 0), (w, h), (0, 255, 0), 2)
    else:
        # this for draw line on image
        cv2.line(image, coordinatePoint_1, coordinatePoint_2, (0, 255, 0), 1)
    return image


def calculate_ratio_image2label(label, image):
    """
    This function for calculate the initial ratio of the image.

    Returns:
        ratio_x : ratio width between image and ui window.
        ratio_y : ratio height between image and ui window.
        center : find the center image on window user interface.
    """
    h = label.height()
    w = label.width()
    height, width = image.shape[:2]
    ratio_x = width / w
    ratio_y = height / h
    return ratio_x, ratio_y


def cropping_image(image, right, bottom, left, top):
    """
    Cropping image by ratio from every side.

    Args:
        image: Input image
        right: ratio of right side (1-0)
        bottom: ratio of bottom side (1-0)
        left: ratio of left side (0-1)
        top: ratio of top side (0-1)

    Returns:
        image has already cropping

    """
    a_right = round(image.shape[1] * right)
    a_bottom = round(image.shape[0] * bottom)
    a_left = round(image.shape[1] * left)
    a_top = round(image.shape[0] * top)
    return image[a_top:a_top + a_bottom, a_left:a_left + a_right]

def draw_list_point_with_text(image, coordinate_point, radius=5):
    """
    Drawing point on the image.

    Args:
        image:
        coordinate_point:
        radius:

    Returns:

    """

    if coordinate_point is not None:
        for i in range(len(coordinate_point)):
            image = cv2.putText(image, str(i + 1), tuple(coordinate_point[i]), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (200, 5, 5), 3, cv2.LINE_AA)
            cv2.circle(image, tuple(coordinate_point[i]), radius, (200, 5, 200), 20, -1)
    return image

def init_ori_ratio(label, image):
    """
    Calculate the initial ratio of the image.

    Returns:
        ratio_x : ratio width between image and ui window.
        ratio_y : ratio height between image and ui window.
        center : find the center image on window user interface.
    """
    h = label.height()
    w = label.width()
    height, width = image.shape[:2]
    ratio_x = width / w
    ratio_y = height / h
    return ratio_x, ratio_y
