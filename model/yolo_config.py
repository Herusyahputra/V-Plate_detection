import cv2
import numpy as np
from model.object_detection_functions import object_detection_analysis_with_nms

# Start borrow algorithm from reposirtory GitHub
class Yolo_config(object):
    def __init__(self, parent):
        """
        This function is atribut classs (Read the network layers/components. The YOLO V4 neural network has 379
        components. They consist of convolutional layers (conv), rectifier linear units (relu) etc.:

        Args:
            parent:

        """
        super(Yolo_config, self).__init__()
        self.parent = parent

        self.class_labels_path = "./backend/coco.names"
        self.class_labels_path = "./backend/coco.names"
        self.class_labels = open(self.class_labels_path).read().strip().split("\n")
        self.class_colors = np.random.randint(0, 255, size=(len(self.class_labels), 4), dtype="uint8")

        self.scalefactor = 1.0 / 255.0
        self.new_size = (316, 316)

        self.score_threshold = 0.4
        self.nms_threshold = 0.4

        # Load model
        self.yolo_model = cv2.dnn.readNetFromDarknet(
            './backend/yolov4-obj.cfg',
            './backend/custom.weights')

        self.model_layers = self.yolo_model.getLayerNames()
        # Loop through all network layers to find the output layers
        self.output_layers = [self.model_layers[model_layer[0] - 1] for model_layer in self.yolo_model.getUnconnectedOutLayers()]

    def detect_using_yolo(self, frame):
        """
        This function is for input pre-processed blob into the model, compute the forward pass for the input,
        storing the results per output layer in a list, compute the forward pass for the input, storing the results
        per output layer in a list, get  the object detections drawn on  the frame

        Args:
            frame:
                image

        Returns:
            frame, winner_boxes, predicted_class_label
        """
        blob = cv2.dnn.blobFromImage(frame, self.scalefactor, self.new_size, swapRB=True, crop=False)
        self.yolo_model.setInput(blob)
        print(self.scalefactor)
        obj_detections_in_layers = self.yolo_model.forward(self.output_layers)
        frame, winner_boxes, predicted_class_label = object_detection_analysis_with_nms(frame, self.class_labels, self.class_colors,
                                                                 obj_detections_in_layers, self.score_threshold,
                                                                 self.nms_threshold)
        print("frame : ", winner_boxes)

        # if running outside Colab notebooks use:
        if not winner_boxes:
            winner_boxes = None

        return frame, winner_boxes, predicted_class_label

