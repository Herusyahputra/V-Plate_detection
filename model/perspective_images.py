import numpy as np
import cv2


class Perspective(object):
    def __init__(self, parent):
        super(Perspective, self).__init__()
        self.parent = parent
        self.image = None
        print("from recognize")

    def perspective_images(self, image):
        # read input
        self.image =image

        # specify desired output size
        width = 600
        height = 400

        # specify conjugate x,y coordinates (not y,x)
        input = np.float32([[62,71], [418,59], [442,443], [29,438]])
        output = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])

        # compute perspective matrix
        matrix = cv2.getPerspectiveTransform(input,output)

        print(matrix.shape)
        print(matrix)

        # do perspective transformation setting area outside input to black
        imgOutput = cv2.warpPerspective(self.image, matrix, (width,height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        print(imgOutput.shape)

        # save the warped output
        cv2.imwrite("image.jpg", imgOutput)
        return imgOutput

