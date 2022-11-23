import cv2
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from .sourceIcon import ResourceIcon


class VideoController(object):
    def __init__(self, parent):
        super(VideoController, self).__init__()
        self.parent = parent
        self.__rs = ResourceIcon()
        self.fps = None
        self.pos_frame = None
        self.__frame_count = None
        self.__minute = None
        self.__minutes = None
        self.__seconds = None
        self.__sec = None
        self.play = False
        self.__playPauseBtn = None
        self.frameNumber = 1
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrame)

    # migrate to video controller
    def nextFrame(self):
        """
        looping the frame showing in label user interface.

        """
        if self.parent.cap:
            success, self.parent.image = self.parent.cap.read()
            if success:
                self.fps = self.parent.cap.get(cv2.CAP_PROP_FPS)
                self.pos_frame = self.parent.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.__frame_count = float(self.parent.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.__showToLabel()
                
            else:
                self.pauseVideo()

        else:
            self.__frame_count = len(os.listdir(self.parent.folderOdometry))
            self.fps = 10
            if self.frameNumber < self.__frame_count-1:
                self.pos_frame = self.frameNumber
                file = self.parent.folderOdometry + "/" + str(self.frameNumber) + ".png"
                self.parent.image = cv2.imread(file)
                self.parent.h, self.parent.w = self.parent.image.shape[:2]
                self.__showToLabel()
                self.frameNumber += 1
            else:
                self.pauseVideo()
        try:
            if self.parent.video_writer is not None:
                image = self.parent.image if self.parent.normal_view else self.parent.result_image
                self.parent.video_writer.write(image)
        except:
            pass

    def __showToLabel(self):
        duration_sec = int(self.__frame_count / self.fps)
        self.__minutes = duration_sec // 60
        duration_sec %= 60
        self.__seconds = duration_sec
        sec_pos = int(self.pos_frame / self.fps)
        self.__minute = int(sec_pos // 60)
        sec_pos %= 60
        self.__sec = sec_pos
        self.__controller()
        self.parent.showToWindow()

    def playPauseVideo(self, btnName):
        """
        Control the play and pause video controller button
        for example, if play is true: it will change the icon button

        """
        if not btnName:
            btnName = None
        self.__playPauseBtn = btnName
        if self.play:
            self.pauseVideo()

        else:
            self.playVideo()

    def pauseVideo(self):
        """
        Pause the frame in video or camera mode.

        """
        try:
            self.timer.stop()
            self.play = False
            if self.__playPauseBtn is not None:
                self.__playPauseBtn.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(self.__rs.iconPlay())))

        except:
            QtWidgets.QMessageBox.warning(
                None,
                "Warning!!",
                "No Source Media found !!!")

    def playVideo(self):
        """
        Play video by connect to timer function.

        """
        try:
            self.timer.start(1000 / self.fps)
            self.play = True
            if self.__playPauseBtn is not None:
                self.__playPauseBtn.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(self.__rs.iconPause())))

        except:
            QtWidgets.QMessageBox.warning(
                None,
                "Warning!!",
                "No Source Media found !!!")

    def __controller(self):
        """
        Manage the video to setup the current timer.

        """
        try:
            dst_value = self.pos_frame * (self.parent.slider_Video.maximum() + 1) / self.__frame_count
            self.parent.slider_Video.blockSignals(True)
            self.parent.slider_Video.setValue(dst_value)
            self.parent.slider_Video.blockSignals(False)
            current = self.parent.label_time_recent
            current.setAlignment(QtCore.Qt.AlignCenter)
            current.setText("%02d : %02d" % (self.__minute, self.__sec))

            if self.__minutes < 0:
                "this is for live camera, when the times more than 1000 minutes, it will set to 00:00"
                my_label3 = self.parent.label_time_end
                my_label3.setAlignment(QtCore.Qt.AlignCenter)
                my_label3.setText("00 : 00")

            else:
                my_label3 = self.parent.label_time_end
                my_label3.setAlignment(QtCore.Qt.AlignCenter)
                my_label3.setText("%02d : %02d" %
                                  (self.__minutes, self.__seconds))

        except:
            pass

    def stopVideo(self):
        """
        Stop video and set the time as a beginning, including the slider time

        """
        try:
            self.play = False
            if self.__playPauseBtn is not None:
                self.__playPauseBtn.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(self.__rs.iconPlay())))
            if self.parent.cap:
                self.parent.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.frameNumber = 1
            self.nextFrame()
            self.__reset_label_time()
            self.timer.stop()

        except:
            QtWidgets.QMessageBox.warning(
                None,
                "Warning!!",
                "No Source Media found !!!")

    def __reset_label_time(self):
        """
        Reset the time when open the new video.

        """
        current = self.parent.label_time_recent
        current.setAlignment(QtCore.Qt.AlignCenter)
        current.setText("00 : 00")

        my_label3 = self.parent.label_time_end
        my_label3.setAlignment(QtCore.Qt.AlignCenter)
        my_label3.setText("%02d : %02d" %
                          (self.__minutes, self.__seconds))

    def rewindVideo(self):
        """
        Previous video is 5 seconds.

        """
        try:
            if self.parent.cap:
                position = self.pos_frame - 5 * self.fps
                self.parent.cap.set(cv2.CAP_PROP_POS_FRAMES, position)

            else:
                self.frameNumber -= 5 * self.fps
                if self.frameNumber <= 1:
                    self.frameNumber = 1
            self.nextFrame()

        except:
            QtWidgets.QMessageBox.warning(
                None,
                "Warning!!",
                "No Source Media found !!!")

    def forwardVideo(self):
        """
        skip video in 5 seconds.

        """
        try:
            if self.parent.cap:
                position = self.pos_frame + 5 * self.fps
                self.parent.cap.set(cv2.CAP_PROP_POS_FRAMES, position)

            else:
                self.frameNumber += 5 * self.fps
                if self.frameNumber >= self.__frame_count:
                    self.frameNumber = self.__frame_count

            self.nextFrame()

        except:
            QtWidgets.QMessageBox.warning(
                None,
                "Warning!!",
                "No Source Media found !!!")

    def sliderController(self, value):
        """
        Set and control the slider to control the video.

        Args:
            value (): Value from slider

        """
        try:
            dst_frame = self.__frame_count * value / self.parent.slider_Video.maximum()
            if self.parent.cap:
                self.parent.cap.set(cv2.CAP_PROP_POS_FRAMES, dst_frame)

            else:
                self.frameNumber = round(dst_frame)
                if self.frameNumber <= 1:
                    self.frameNumber = 1

            self.nextFrame()
            self.pauseVideo()
            self.play = False

        except:
            QtWidgets.QMessageBox.warning(
                None,
                "Warning!!",
                "No Source Media found !!!")
