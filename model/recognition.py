import cv2
import imutils
import pytesseract

class Recognition(object):
    def __init__(self, parent):
        super(Recognition, self).__init__()
        self.parent = parent
        self.image = None
        print("from recognize")

    def recognition_character(self, image):
        self.image = image
        image = imutils.resize(self.image, width=300 )
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
        edged = cv2.Canny(gray_image, 30, 200)
        cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image1=self.image.copy()
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
        screenCnt = None
        image2 = self.image.copy()

        i=7
        for c in cnts:
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.1 * perimeter, True)
                if len(approx) == 4:
                        screenCnt = approx

                        x,y,w,h = cv2.boundingRect(c)
                        new_img=self.image[y:y+h,x:x+w]
                        cv2.imwrite('./'+str(i)+'.png',new_img)
                        i+=1
                        break

        Cropped_loc = './7.png'
        plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
        print("Number plate is:", plate)
        return plate
