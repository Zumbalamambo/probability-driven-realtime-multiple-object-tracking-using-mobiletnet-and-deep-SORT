# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:34:04 2018

@author: Administrator
"""

from .detector import Detector
import configparser
import cv2
import numpy as np

class Mobilenet_Ssd(Detector):
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        model = config.get('mobilenet_ssd', 'model')
        prototxt = config.get('mobilenet_ssd', 'prototxt')

        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.image_width = int(config.get('mobilenet_ssd', 'image_width'))
        self.image_height = int(config.get('mobilenet_ssd', 'image_height'))
        self.confident_threshold = float(config.get('mobilenet_ssd', 'confident_threshold'))
        self.detect_classes = config.get('mobilenet_ssd', 'detect_classes').split(',')
        self.ignore_classes = set(config.get('mobilenet_ssd', 'ignore_classes').split(','))

        self.is_display = config.get('common_config', 'is_display') == 'True'

    def _detect_from_image(self, image_frame):
        blob = cv2.dnn.blobFromImage(image_frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections

    def detect_image(self, image_path):
        image_frame = cv2.imread(image_path)
        (h, w) = image_frame.shape[:2]
        detections_result = self._detect_from_image(cv2.resize(image_frame, (self.image_height, self.image_width)))

        print(detections_result)
        for i in range(detections_result.shape[2]):
            confidence = detections_result[0, 0, i, 2]

            if confidence > self.confident_threshold:
                idx = int(detections_result[0, 0, i, 1])

                if(self.detect_classes[idx] in self.ignore_classes):
                    continue
                else:
                    bounding_box = detections_result[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = bounding_box.astype("int")

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(self.detect_classes[idx], confidence * 100)
                    cv2.rectangle(image_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image_frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) , 2)

        if(self.is_display):
            cv2.imshow('image', image_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_video(self, video_path):
        pass

    def detect_webcam(self):
        pass