# -*- coding: utf-8 -*-

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import configparser
import cv2
import numpy as np

class Mobilenetv2_Ssdlite(object):
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        model_path = config.get('mobilenetv2_ssdlite', 'model_path')

        self.detect_classes = config.get('mobilenetv2_ssdlite', 'detect_classes').split(',')
        self.ignore_classes = set(config.get('mobilenetv2_ssdlite', 'ignore_classes').split(','))
        self.detection_graph = self._read_model_from_pb_file(model_path)
        self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

    def _read_model_from_pb_file(self, model_path):
      detection_graph = tf.Graph()
      with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')
      return detection_graph

    def detect_image(self, image_frame, height, width, to_xywh, confident_threshold):
        detection_results = []
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_frame_expanded = np.expand_dims(image_frame, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        bounding_box = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (bounding_box, scores, classes, num_detections) = self.sess.run(
            [bounding_box, scores, classes, num_detections],
            feed_dict={image_tensor: image_frame_expanded})
        for i in range(int(num_detections[0])):
            if(scores[0][i] > confident_threshold):
              if(self.detect_classes[int(classes[0][i])] in self.ignore_classes):
                  continue
              else:
                  ret1, ret2, ret3, ret4 = bounding_box[0][i][0] * height, bounding_box[0][i][1] * width, bounding_box[0][i][2] * height, bounding_box[0][i][3] * width

                  if(to_xywh is True):
                      ret1, ret2, ret3, ret4 = int(ret1), int(ret2), int(ret3-ret1), int(ret4-ret2)

                      if ret1 < 0 :
                          ret3 += ret1
                          ret1 = 0
                      if ret2 < 0 :
                          ret4 += ret2
                          ret2 = 0

                  detection_results.append([self.detect_classes[int(classes[0][i])], ret2, ret1, ret4, ret3, scores[0][i]])
        return detection_results


if __name__ == '__main__':
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(MODEL_NAME, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    cap = cv2.VideoCapture('../_samples/MOT17-09-FRCNN.mp4')
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        while True:
          start_time = time.time()
          ret, image_np = cap.read()
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          # @ todo add
          cv2.imshow('object detection', image_np)
          print(1 / (time.time()-start_time))
          if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break