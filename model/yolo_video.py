# import the relevant libraries
import numpy as np
import cv2  # openCV

class_labels_path = "/home/moil-dev002/Documents/Plate_Detection/model/coco.names"
class_labels = open(class_labels_path).read().strip().split("\n")
# print(class_labels)

# declare repeating bounding box colors for each class
# 1st: create a list colors as an RGB string array
# Example: Red, Green, Blue, Yellow, Magenda
class_colors = ["255,0,0", "0,255,0", "0,0,255", "255,255,0", "255,0, 255"]

# 2nd: split the array on comma-seperated strings and for change each string type to integer
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]

# 3d: convert the array or arrays to a numpy array
class_colors = np.array(class_colors)

# 4th: tile this to get 80 class colors, i.e. as many as the classes  (16rows of 5cols each).
# If you want unique colors for each class you may randomize the color generation
# or set them manually
class_colors = np.tile(class_colors, (16, 5))
class_colors = np.random.randint(0, 255, size=(len(class_labels), 5), dtype="uint8")

# for the image2blob conversion
scalefactor = 1.0 / 255.0
new_size = (416, 416)

# for the NMS
score_threshold = 0.5
nms_threshold = 0.4

# Load the pre-trained model
yolo_model = cv2.dnn.readNetFromDarknet('/home/moil-dev002/Documents/Plate_Detection/backend/yolov4.cfg', '/home/moil-dev002/Documents/Plate_Detection/backend/yolov4.weights')

# Read the network layers/components. The YOLO V4 neural network has 379 components.
# They consist of convolutional layers (conv), rectifier linear units (relu) etc.:
model_layers = yolo_model.getLayerNames()

# Loop through all network layers to find the output layers
output_layers = [model_layers[model_layer[0] - 1] for model_layer in yolo_model.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://192.168.103.211:8000/stream.mjpg")
# cap = cv2.VideoCapture("/home/moil-dev002/Documents/Plate_Detection/Videos/Recorded_042016_2944.avi")
new_width = 640
new_height = 480
dim = (new_width, new_height)

from object_detection_functions import object_detection_analysis_with_nms
if cap.isOpened():
    while True:
        # get the current frame from video stream
        ret, frame = cap.read()
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(frame, scalefactor, new_size, swapRB=True, crop=False)

        # input pre-processed blob into the model
        yolo_model.setInput(blob)
        # compute the forward pass for the input, storing the results per output layer in a list
        obj_detections_in_layers = yolo_model.forward(output_layers)
        # get  the object detections drawn on  the frame
        frame, winner_boxes, _ = object_detection_analysis_with_nms(frame, class_labels, class_colors,
                                                                 obj_detections_in_layers, score_threshold,
                                                                 nms_threshold)

        # display the frame
        # cv2_imshow(frame)
        # if running outside Colab notebooks use:
        cv2.imshow("frame", frame)

        # terminate while loop if 'q' key is pressed - applicable outside the notebooks
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # releasing the stream and the camera
    cap.release()
    cv2.destroyAllWindows()
