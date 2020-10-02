#!/usr/bin/env python3
"""Program that uses the Yolo v3 algorithm to perform object detection"""

import tensorflow.keras as K


class Yolo:
    """Class Yolov3 to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class Constructor"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
