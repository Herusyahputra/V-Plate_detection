# from . import exif


class MetaImage(object):
    """
    Open an image based on the file path. Read and write the metadata of the image.
    This Library just provide can modify the comment, read the comment and clear the comment in metadata.
    """

    def __init__(self, filename, encoding='utf-8'):
        """
        Open an image and load its metadata.

        Args:
            filename (): The name of image data.
            encoding (): Convert string to data that can stored on metadata image
        """
        self.img = exif.Image(filename.encode(encoding))

    def close(self):
        """
        Free the memory or close from accessing image data.

        Returns:
            None
        """
        self.img.close_image()

        # Disable all methods and properties
        def closed_warning():
            raise RuntimeError('The image has been closed, so it is not allowed to operate.')

        for attr in dir(self):
            if not attr.startswith('__'):
                if callable(getattr(self, attr)):
                    setattr(self, attr, closed_warning)
                else:
                    setattr(self, attr, None)

    def read_comment(self, encoding='utf-8') -> str:
        """
        This function is for read the comment from metadata Image and return with
        decode the data to be string.

        Args:
            encoding (): Convert string to data that can stored on metadata image

        Returns:
            The comment data from Metadata Image.
        """
        return self.img.read_comment().decode(encoding)

    def modify_comment(self, data: str, encoding='utf-8'):
        """
        Modify the comment in metadata Image.

        Args:
            data (): the comment string.
            encoding (): Convert string to data that can stored on metadata image

        Returns:
            None
        """
        self.img.modify_comment(data, encoding)

    def clear_comment(self):
        """
        Clear all the comment in metadata Image.

        Returns:
            None.
        """
        self.img.clear_comment()
