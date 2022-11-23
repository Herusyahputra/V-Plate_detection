

class DataProperties:
    def __init__(self):
        """
        The backend that contains all the data logic.
        The model's job is to simply manage the data. Whether the data is from a database,
        API, or a JSON object, the model is responsible for managing it.

        """
        super(DataProperties, self).__init__()
        self.__image_original = None
        self.__image_drawing = None
        self.__image_result = None
        self.__source_path = None
        self.__mode_view = "Fisheye"
        self.__properties_anypoint = {"alpha": 0, "beta": 0, "roll": 0, "zoom": 2, "mode": 1}
        self.__properties_panorama = {"alpha_max": 110, "alpha_min": 10}
        self.__properties_video = {"video": False, "pos_frame": 0, "frame_count": 0, "total_minute": 0,
                                   "total_second": 0, "current_minute": 0, "current_second": 0}

    # -------------------------------------- image ------------------------------------------
    @property
    def image_original(self):
        """
            This function is for get image original
        Returns:
            image original
        """
        return self.__image_original

    @image_original.setter
    def image_original(self, value):
        """
            This function is for set image original
        Args:
            value: image original
        Returns:
            None
        """
        self.__image_original = value

    @property
    def image_drawing(self):
        """
            This function is for get image drawing
        Returns:
            image original
        """
        return self.__image_drawing

    @image_drawing.setter
    def image_drawing(self, value):
        """
            This function is for set image drawing
        Args:
            value: image drawing
        Returns:
            None
        """
        self.__image_drawing = value

    @property
    def image_result(self):
        """
            This function is for getter image result (can be anypoint, original, panorama etc)
        Returns:
            image result
        """
        return self.__image_result

    @image_result.setter
    def image_result(self, value):
        """
            This function is for set image result
        Args:
            value: image result

        Returns:
            None
        """
        self.__image_result = value

    # -------------------------------------- properties ------------------------------------------
    @property
    def properties_anypoint(self):
        """
            This function is for get properties anypoint
        Returns:
            properties anypoint
        """
        return self.__properties_anypoint

    @properties_anypoint.setter
    def properties_anypoint(self, value):
        """
            This function us for set properties anypoint
        Args:
            value: properties anypoint
        Returns:
            None
        """
        self.__properties_anypoint = value

    @property
    def properties_panorama(self):
        """
            This function is for get properties panorama
        Returns:
            properties panorama
        """
        return self.__properties_panorama

    @properties_panorama.setter
    def properties_panorama(self, value):
        """
            This function us for set properties panorama
        Args:
            value: properties panorama
        Returns:
            None
        """
        self.__properties_panorama = value

    @property
    def properties_video(self):
        """
            This function is for get properties video
        Returns:
            properties video
        """
        return self.__properties_video

    @properties_video.setter
    def properties_video(self, value):
        """
            This function us for set properties video
        Args:
            value: properties video
        Returns:
            None
        """
        self.__properties_video = value

    # -------------------------------------- other ------------------------------------------
    @property
    def source_path(self):
        """
            this function is for get object source_path
        Returns:
            none
        """
        return self.__source_path

    @source_path.setter
    def source_path(self, value):
        """
            this function is for set source_path object
        Args:
            value: source_path object
        Returns:
            None
        """
        self.__source_path = value

    @property
    def mode_view(self):
        """
            this function is for get object __mode_view
        Returns:
            none
        """
        return self.__mode_view

    @mode_view.setter
    def mode_view(self, value):
        """
            this function is for set mode_view object
        Args:
            value: mode_view object
        Returns:
            None
        """
        self.__mode_view = value
