# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:29:25 2018

@author: Administrator
"""
import configparser
import cv2
import time

from PIL import Image

from .detector_template import Detector_Template
from .mobilenet_ssd import Mobilenet_Ssd
from .mobilenetv2_ssdlite import Mobilenetv2_Ssdlite
from .squeezenetv1_0 import Squeezenetv1_0
from .yolo import YOLO

class Detector(Detector_Template):
    def __init__(self, detector_name, config_path):
        self.detector_name = detector_name
        self.detector = self._detector_selection(detector_name, config_path)

        config = configparser.ConfigParser()
        config.read(config_path)
        self.detect_frequency = int(config.get('common_config', 'detect_frequency'))
        self.is_display = config.get('common_config', 'is_display') == 'True'
        self.confident_threshold = float(config.get('common_config', 'confident_threshold'))

    def _detector_selection(self, detector_name, config_path):
        detector_map = {'mobilenet_ssd' : Mobilenet_Ssd,
                        'mobilenetv2_ssdlite' : Mobilenetv2_Ssdlite,
                        'squeezenetv1_0' : Squeezenetv1_0,
                        'yolo' : YOLO,
        }

        return detector_map[detector_name](config_path)

    def _detect_image(self, image_frame, to_xywh):
        height, width = image_frame.shape[:2]
        if(self.detector_name == 'yolo'):
            image_frame = Image.fromarray(image_frame)
        detection_results = self.detector.detect_image(image_frame, height, width, to_xywh, self.confident_threshold)
        return detection_results

    def _detect_and_display_image_sequence(self, cap, to_xywh, is_display):
        total_time = time.time()
        counter = 0
        start_time = time.time()
        step_counter = 0
        while True:
            ret, image_frame = cap.read()
            if ret != True:
                break

            if(step_counter % self.detect_frequency == 0 or counter == 0):
                detection_results = self._detect_image(image_frame, to_xywh)

            if(is_display is True):
                self._display(image_frame, detection_results)
                counter += 1
                step_counter += 1
                if(step_counter % self.detect_frequency == 0 or counter == 0):
                    end_time = time.time()
                    cv2.putText(image_frame, 'FPS:' + str(round(step_counter / (end_time - start_time), 1)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2)
                    start_time = time.time()
                    step_counter = 0

                cv2.imshow('image', image_frame)

                if cv2.waitKey(1) >= 0:
                    break

        print('Average FPS:', round(counter / (time.time() - total_time), 1))
        print('Total eplased:', round(time.time() - total_time, 2))
        cap.release()
        cv2.destroyAllWindows()

    def _display(self, image_frame, results):
        for result in results:
            confident = result[5]
            label = "{}: {:.2f}%".format(result[0], confident*100)
            cv2.rectangle(image_frame, (result[1], result[2]), (result[3], result[4]), (255, 0, 0), 2)
            y = result[2] - 15 if  result[2] - 15 > 15 else  result[2] + 15
            cv2.putText(image_frame, label, (result[1], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) , 2)

    def detect_image_frame(self, image_frame, to_xywh):
        detection_results = self._detect_image(image_frame, to_xywh)
        ret_results = list()
        for detection_result in detection_results:
            ret_results.append(detection_result)
        return ret_results

    def detect_image(self, image_path, to_xywh=False, is_display=None):
        image_frame = cv2.imread(image_path)
        detection_results = self._detect_image(image_frame, to_xywh)

        if(is_display is None):
            is_display = self.is_display

        if(is_display is True):
            self._display(image_frame, detection_results)
            cv2.imshow('image', image_frame)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detection_results

    def detect_video(self, video_path, to_xywh=False):
        cap = cv2.VideoCapture(video_path)
        self._detect_and_display_image_sequence(cap, to_xywh, self.is_display)

    def detect_webcam(self, to_xywh=False):
        cap = cv2.VideoCapture(0)
        self._detect_and_display_image_sequence(cap, to_xywh, self.is_display)

    def generate_detecions(self, video_stream, output_file=None, to_xywh=True):
        cap = cv2.VideoCapture(video_stream)
        total_time = time.time()
        counter = 0
        start_time = time.time()
        step_counter = 0
        frame_counter = 1
        result_list = []
        
        DETECT_FREQUENCY = 1
        DOWN_SAMPLE_RATIO = 0.5
        while True:
            ret, image_frame = cap.read()
            if ret != True:
                break
            (h, w) = image_frame.shape[:2]
            image_frame = cv2.resize(image_frame, (int(w * DOWN_SAMPLE_RATIO), int(h * DOWN_SAMPLE_RATIO)))
            if((step_counter % DETECT_FREQUENCY == 0) or counter == 0):
                detection_results = self._detect_image(image_frame, to_xywh)
                step_counter = 0

            for detection in detection_results:
                result_list.append([frame_counter, detection[1], detection[2], detection[3], detection[4], detection[5]])
            frame_counter += 1
            counter += 1

        print('Total eplased:', round(time.time() - total_time, 2))
        try:
            with open(output_file + '_det.txt', 'w') as f:
                for result in result_list:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,%2f,-1,-1,-1' % (int(result[0]), -1, float(result[1]), float(result[2]), float(result[3]), float(result[4]), float(result[5])),file=f)
        except Exception as e:
            print(e)
            print('Something went wrong when writing output file!')

        cap.release()
        cv2.destroyAllWindows()