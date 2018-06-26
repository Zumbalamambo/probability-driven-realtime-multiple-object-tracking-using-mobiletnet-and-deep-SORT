# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:34:04 2018

@author: Administrator
"""

from .detector import Detector
import configparser
import cv2
import numpy as np
import time

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
        cap = cv2.VideoCapture(video_path)
        total_time = time.time()
        counter = 0
        start_time = time.time()
        step_counter = 0
        while True:
            try:
                ret, image_frame = cap.read()
                (h, w) = image_frame.shape[:2]
            except:
                break
            if(counter == 0 or step_counter % 2 == 0):
                step_counter = 0
                detections_result = self._detect_from_image(cv2.resize(image_frame, (self.image_height, self.image_width)))

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

            counter += 1
            step_counter += 1
            if(counter == 0 or step_counter % 2 == 0):
                end_time = time.time()
                cv2.putText(image_frame, 'FPS:' + str(round(step_counter / (end_time - start_time), 1)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2)
                start_time = time.time()

            if(self.is_display):
                cv2.imshow('image', image_frame)

            # ESC to break
            if cv2.waitKey(1) >= 0:
                break

        print('Average FPS:', round(counter / (time.time() - total_time), 1))
        print('Total eplased:', round(time.time() - total_time, 2))
        cap.release()
        cv2.destroyAllWindows()

    def detect_webcam(self):
        cap = cv2.VideoCapture(0)

        total_time = time.time()
        counter = 0

        while True:
            start_time = time.time()
            ret, image_frame = cap.read()
            (h, w) = image_frame.shape[:2]
            detections_result = self._detect_from_image(cv2.resize(image_frame, (self.image_height, self.image_width)))

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

            counter += 1
            end_time = time.time()
            cv2.putText(image_frame, 'FPS:' + str(round(1.0 / (end_time - start_time), 1)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2)

            if(self.is_display):
                cv2.imshow('image', image_frame)

            # ESC to break
            if cv2.waitKey(1) >= 0:
                break

        print('Average FPS:', round(counter / (time.time() - total_time), 1))

        cap.release()
        cv2.destroyAllWindows()

    def detect_box_from_image(self, image_frame):
        #(h, w) = image_frame.shape[:2]
        (w, h) = image_frame.shape[:2]
        detections_result = self._detect_from_image(cv2.resize(image_frame, (self.image_height, self.image_width)))

        return_boxs = []
        for i in range(detections_result.shape[2]):
            confidence = detections_result[0, 0, i, 2]
            if confidence > self.confident_threshold:
                idx = int(detections_result[0, 0, i, 1])

                if(self.detect_classes[idx] in self.ignore_classes):
                    continue
                else:
                    #bounding_box = detections_result[0, 0, i, 3:7] * np.array([w, h, w, h])
                    bounding_box = detections_result[0, 0, i, 3:7] * np.array([h, w, h, w])
                    (startX, startY, endX, endY) = bounding_box.astype("int")
                    x = int(startX)
                    y = int(startY)
                    w = int(endX - startX)
                    h = int(endY - startY)
                    """
                    if x < 0 :
                        w = w + x
                        x = 0
                    if y < 0 :
                        h = h + y
                        y = 0
                    """
                    #if(x <= 10 or y <= 10 or w <= 10 or h <= 10):
                    #    continue

                    return_boxs.append([x,y,w,h])

        return return_boxs

    def detect_box_from_image_2(self, image_frame):
        (height, width) = image_frame.shape[:2]
        detections_result = self._detect_from_image(cv2.resize(image_frame, (self.image_height, self.image_width)))

        return_boxs = []
        for i in range(detections_result.shape[2]):
            confidence = detections_result[0, 0, i, 2]

            if confidence > self.confident_threshold:
                idx = int(detections_result[0, 0, i, 1])

                if(self.detect_classes[idx] in self.ignore_classes):
                    continue
                else:
                    bounding_box = detections_result[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = bounding_box.astype("int")
                    x = int(startX)
                    y = int(startY)
                    w = int(endX - startX)
                    h = int(endY - startY)
                    return_boxs.append([x, y, w, h])

        return return_boxs