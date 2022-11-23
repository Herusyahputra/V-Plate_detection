from PyQt6 import QtWidgets, QtGui, QtCore
import cv2
import os
import shutil
import numpy as np
import datetime
import math
import json
from Moildev import Moildev
from .exif_lib import MetaImage
from .camera_source import CameraSource
from .camera_parameter import CameraParameters


class MoilUtils(object):
    __camera_params = "utils/camera_parameters.json"
    __camera_type = None

    def __init__(self):
        super(MoilUtils, self).__init__()

    @classmethod
    def selectCameraSource(cls):
        openCamSource = QtWidgets.QDialog()
        winOpenCam = CameraSource(openCamSource)
        openCamSource.exec_()
        return winOpenCam.camera_source

    @classmethod
    def parametersForm(cls):
        """
        Open the window of camera parameter form, this window you can update, add, and
        delete the camera parameter from database.

        """
        camParam = QtWidgets.QDialog()
        CameraParameters(camParam, cls.__camera_params)
        camParam.exec_()

    @classmethod
    def createMapsMultipleView(cls, numberOfView, typeCamera, listAlpha, listBeta, listZoom, mode=2):
        moildev = []
        mapsX = []
        mapsY = []
        for i in range(numberOfView):
            moildev.append(cls.connectToMoildev(typeCamera))
            mapX, mapY = moildev[i].getAnypointMaps(listAlpha[i],
                                                    listBeta[i],
                                                    listZoom[i],
                                                    mode)
            mapsX.append(mapX)
            mapsY.append(mapY)
        return mapsX, mapsY

    @classmethod
    def selectCameraType(cls):
        """
        Select the camera type prompt.

        """
        cls.__camera_type = None
        with open(cls.__camera_params) as f:
            data = json.load(f)
        new_list = []
        for key in data.keys():
            new_list.append(key)
        Dialog = QtWidgets.QDialog()
        Dialog.setObjectName("Dialog")
        Dialog.setWindowTitle("Select Camera !!!")
        Dialog.resize(240, 120)
        buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        buttonBox.setGeometry(QtCore.QRect(20, 80, 200, 32))
        buttonBox.setOrientation(QtCore.Qt.Horizontal)
        buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        buttonBox.setObjectName("buttonBox")
        comboBox_cam_type = QtWidgets.QComboBox(Dialog)
        comboBox_cam_type.setGeometry(QtCore.QRect(20, 40, 200, 30))
        comboBox_cam_type.setObjectName("comboBox")
        comboBox_cam_type.addItems(new_list)
        label = QtWidgets.QLabel(Dialog)
        label.setGeometry(QtCore.QRect(10, 10, 220, 30))
        font = QtGui.QFont()
        font.setFamily("DejaVu Serif")
        font.setPointSize(13)
        label.setFont(font)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setObjectName("label")
        label.setText("Select the camera type !!!")
        buttonBox.accepted.connect(lambda: cls.__accept_btn(Dialog, comboBox_cam_type))
        buttonBox.rejected.connect(lambda: cls.__reject_btn(Dialog))
        Dialog.exec_()
        return cls.__camera_type

    @classmethod
    def __accept_btn(cls, dialog, msg):
        dialog.accept()
        cls.__camera_type = msg.currentText()

    @classmethod
    def __reject_btn(cls, dialog):
        dialog.reject()
        cls.__camera_type = None

    @classmethod
    def selectFile(cls, parent, title, dir_path, file_filter):
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
        options = QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, dir_path,
                                                             file_filter,
                                                             options=options)
        return file_path

    @classmethod
    def selectDirectory(cls, parent=None):
        """
        Select directory to save image. This function create to make it not always ask the directory by open dialog,
        after directory save not None, it will pass open dialog prompt.

        Returns:
            None.
        """
        option = QtWidgets.QFileDialog.DontUseNativeDialog
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            parent, 'Select your folder', options=option)
        return directory

    @classmethod
    def copyDirectory(cls, original, destination):
        """
        Copy directory.

        Args:
            original (): source path or original path folder
            destination (): destination directory.

        Returns:
            file copied in destination directory.
        """
        directoryName = os.path.basename(original)
        destinationPath = os.path.join(destination, directoryName)
        shutil.copytree(original, destinationPath)

    @classmethod
    def drawPolygon(cls, image, mapX, mapY):
        """
        Draw polygon from mapX and mapY given in the original image.

        Args:
            image: Original image
            mapX: map image X from anypoint process
            mapY: map image Y from anypoint process

        return:
            image:
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

    @classmethod
    def resizeImage(cls, image, width):
        """
        Resize image original with our size we want

        Args:
            image (): image original
            width (): image width we want

        Returns:
            result: image has been resize

        """
        h, w = image.shape[:2]
        r = width / float(w)
        hi = round(h * r)
        result = cv2.resize(image, (width, hi),
                            interpolation=cv2.INTER_AREA)
        return result

    @classmethod
    def readCameraType(cls, image_file):
        """
        Read the camera used from metadata image.

        Args:
            image_file ():

        Returns:

        """
        img = MetaImage(image_file)
        camera_type = img.read_comment()
        img.close()
        return camera_type

    @classmethod
    def writeCameraType(cls, image_file, typeCamera):
        """
        Read the camera used from metadata image.

        Args:
            image_file ():
            typeCamera ():

        Returns:

        """
        img = MetaImage(image_file)
        img.modify_comment(typeCamera)
        img.close()

    @classmethod
    def saveImage(cls, image, dst_directory=None, name=None, type_camera=None):
        """
        saved image
        Args:
            image ():
            dst_directory (): destination directory
            name ():
            type_camera ():

        Returns:

        """
        ss = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        if name is None:
            name = dst_directory + "/" + str(ss) + "_" + ".png"
        else:
            name = dst_directory + "/" + str(ss) + "_" + str(name) + ".png"
        cv2.imwrite(name, image)
        if type_camera is not None:
            img = MetaImage(name)
            img.modify_comment(type_camera)
            img.close()
        return ss

    @staticmethod
    def calculateHeight(image, width):
        """
        Calculate the height image with the same ratio with the size original image.

        Args:
            image: original image
            width: size image we want

        Returns:
            height: height image
        """

        h, w = image.shape[:2]
        r = width / float(w)
        height = round(h * r)
        return height

    @classmethod
    def readImage(cls, image_path):
        """
        Reading the image from given file path using openCV.

        Args:
            image_path : The path of image file

        return:
            Image: load image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("`{}` cannot be loaded".format(image_path))
        return image

    @classmethod
    def rotateImage(cls, src, angle, center=None, scale=1.0):
        """
        Turn an image in a clockwise or counterclockwise direction depend on angle value given.

        Args:
            src: original image
            angle: the value angle for turn the image
            center: determine the specific coordinate to rotate image
            scale: scale image

        Returns:
            dst image: rotated image
        """
        h, w = src.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(src, m, (w, h))
        return rotated

    @classmethod
    def connectToMoildev(cls, type_camera, parent=None):
        """
        Connect to Moildev SDK, need provide camera parameter database and type of camera.

        Args:
            type_camera ():
            parent ():

        Returns:

        """
        if type_camera:
            moildev = Moildev(cls.__camera_params, type_camera)

        else:
            QtWidgets.QMessageBox.warning(
                parent,
                "Warning!!",
                "The image not support for this application, \n\nPlease contact developer!!")
            print("the image not support for this application, please contact developer!!")
            moildev = None

        return moildev

    @classmethod
    def showImageToLabel(cls, label, image, width, angle=0, plusIcon=False):
        """
        Showing image to the window in user interface.

        Args:
            plusIcon ():
            label ():
            image ():
            width ():
            angle ():

        Returns:

        """

        height = cls.calculateHeight(image, width)
        image = cls.resizeImage(image, width)
        image = MoilUtils.rotateImage(image, angle)
        if plusIcon:
            # draw plus icon on image and show to label
            h, w = image.shape[:2]
            w1 = round((w / 2) - 10)
            h1 = round(h / 2)
            w2 = round((w / 2) + 10)
            h2 = round(h / 2)
            w3 = round(w / 2)
            h3 = round((h / 2) - 10)
            w4 = round(w / 2)
            h4 = round((h / 2)) + 10
            cv2.line(image, (w1, h1), (w2, h2), (0, 255, 0), 2)
            cv2.line(image, (w3, h3), (w4, h4), (0, 255, 0), 2)

        label.setMinimumSize(QtCore.QSize(width, height))
        label.setMaximumSize(QtCore.QSize(width, height))
        image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                             QtGui.QImage.Format.Format_RGB888).rgbSwapped()
        label.setPixmap(QtGui.QPixmap.fromImage(image))

    @staticmethod
    def remap(image, mapX, mapY):
        image = cv2.remap(image, mapX, mapY, cv2.INTER_CUBIC)
        return image

    @staticmethod
    def drawRectangle(image, point_1, point_2, thickness=5):
        image = cv2.rectangle(image, point_1, point_2, (0, 0, 225), thickness)
        return image

    @classmethod
    def cornerDetection(cls, image, sigma=3, threshold=0.01):
        """
        Corner detection using haris corner detection method. it will return list of corner for every point.

        Args:
            image (): the image source
            sigma ():
            threshold ():

        Returns:

        """
        # height, width = image.shape
        # shape = (height, width)
        # Calculate the dx,dy gradients of the image (np.gradient doesnt work)
        image = MoilUtils.convertToGray(image)
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        # Get angle for rotation
        _, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)
        # Square the derivatives (A,B,C of H) and apply apply gaussian filters to each
        sigma = (sigma, sigma)
        Ixx = cv2.GaussianBlur(dx * dx, sigma, 0)
        Ixy = cv2.GaussianBlur(dx * dy, sigma, 0)
        Iyy = cv2.GaussianBlur(dy * dy, sigma, 0)

        H = np.array([[Ixx, Ixy], [Ixy, Iyy]])
        # Find the determinate
        num = (H[0, 0] * H[1, 1]) - (H[0, 1] * H[1, 0])
        # Find the trace
        trace = H[0, 0] + H[1, 1]
        # Find the response using harmonic mean of the eigenvalues (Brown et. al. variation)
        R = np.nan_to_num(num / trace)

        # Adaptive non-maximum suppression, keep the top 1% of values and remove non-maximum points in a 9x9
        # neighbourhood
        R_flat = R[:].flatten()
        # Get number of values in top threshold %
        N = int(len(R_flat) * threshold)
        # Get values in top threshold %
        top_k_percentile = np.partition(R_flat, -N)[-N:]
        # Find lowest value in top threshold %
        minimum = np.min(top_k_percentile)
        # Set all values less than this to 0
        R[R < minimum] = 0
        # Set non-maximum points in an SxS neighbourhood to 0
        s = 9
        for h in range(R.shape[0] - s):
            for w in range(R.shape[1] - s):
                maximum = np.max(R[h:h + s + 1, w:w + s + 1])
                for i in range(h, h + s + 1):
                    for j in range(w, w + s + 1):
                        if R[i, j] != maximum:
                            R[i, j] = 0

        # Return harris corners in [H, W, R] format
        features = list(np.where(R > 0))
        features.append(ang[np.where(R > 0)])
        corners = zip(*features)
        corner_list = cls.__get_corner_list(list(corners))
        return corner_list

    @classmethod
    def drawCorners(cls, image, corners):
        """
        Drawing corner from corner on the image

        Args:
            image ():
            corners ():

        Returns:

        """
        i = 0
        for h, w in corners:
            cv2.circle(image, (w, h), 3, (0, 0, 255), -1)
            # caption = '{},{}'.format(h, w)
            cv2.putText(image, str(i), (w - 5, h + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0))
            i += 1
        return image

    @classmethod
    def __get_corner_list(cls, corners):
        """
        This is private function to reshape corner list to another shape that can use in drawing corner.

        Args:
            corners (): corner point.

        Returns:
            coordinate point for very corner.
        """
        coordinateList = []
        for h, w, r in corners:
            caption = '{},{}'.format(h, w)
            res = tuple(map(int, caption.split(',')))
            coordinateList.append(res)
        return coordinateList

    @classmethod
    def drawPoint(cls, image, coordinatePoint, radius=5):
        """
        Drawing point on the image.

        Args:
            image ():
            coordinatePoint ():
            radius ():

        Returns:

        """

        if coordinatePoint is not None:
            w, h = image.shape[:2]
            if h >= 1000:
                cv2.circle(image, coordinatePoint, radius, (0, 255, 0), 20, -1)
            else:
                cv2.circle(image, coordinatePoint, radius, (0, 255, 0), -1)
        return image

    @classmethod
    def drawLine(cls, image, coordinatePoint_1=None, coordinatePoint_2=None):
        """

        Args:
            image ():
            coordinatePoint_1 ():
            coordinatePoint_2 ():

        Returns:

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

    @classmethod
    def distance(cls, point_a, point_b):
        """
        Returns the distance between two points.

        Args:
            point_a ():
            point_b ():

        Returns:
            distance between 2 point
        """
        x0, y0 = point_a
        x1, y1 = point_b
        return math.fabs(x0 - x1) + math.fabs(y0 - y1)

    @classmethod
    def convertToGray(cls, image):
        """
        Convert rgb to grayscale.

        Args:
            image ():

        Returns:

        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def ratio_image_to_label(label, image):
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


    @classmethod
    def calculateRatioImage(cls, label, image):
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
