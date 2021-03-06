# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:34:04 2018

@author: Administrator
"""

import configparser
import cv2
import numpy as np

class Squeezenetv1_0(object):
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        model = config.get('squeezenetv1_0', 'model')
        prototxt = config.get('squeezenetv1_0', 'prototxt')

        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.image_width = int(config.get('squeezenetv1_0', 'image_width'))
        self.image_height = int(config.get('squeezenetv1_0', 'image_height'))
        mean_substraction = config.get('squeezenetv1_0', 'mean_substraction')
        self.mean_substraction = (int(mean_substraction.split(',')[0]), int(mean_substraction.split(',')[1]), int(mean_substraction.split(',')[2])) 
        self.detect_classes = config.get('squeezenetv1_0', 'detect_classes').split(',')
        self.ignore_classes = set(config.get('squeezenetv1_0', 'ignore_classes').split(','))

    def detect_image(self, image_frame, height, width, to_xywh, confident_threshold):
        blob = cv2.dnn.blobFromImage(cv2.resize(image_frame, (self.image_height, self.image_width)), 1, (self.image_height, self.image_width), (self.mean_substraction[0], self.mean_substraction[1], self.mean_substraction[2]), False)
        self.net.setInput(blob)
        detections = self.net.forward()

        detection_results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])

            if(confidence > confident_threshold):
                if(self.detect_classes[idx] in self.ignore_classes):
                    continue
                else:
                    bounding_box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    ret1, ret2, ret3, ret4 = bounding_box.astype("int")
                    if(to_xywh is True):
                        ret1, ret2, ret3, ret4 = int(ret1), int(ret2), int(ret3-ret1), int(ret4-ret2)

                        if ret1 < 0 :
                            ret3 += ret1
                            ret1 = 0
                        if ret2 < 0 :
                            ret4 += ret2
                            ret2 = 0

                    detection_results.append([self.detect_classes[idx], ret1, ret2, ret3, ret4, confidence])

        return detection_results