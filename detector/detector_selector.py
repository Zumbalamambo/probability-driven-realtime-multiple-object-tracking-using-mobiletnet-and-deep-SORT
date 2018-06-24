# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:29:25 2018

@author: Administrator
"""
from .detector import Detector
from .mobilenet_ssd import Mobilenet_Ssd

class Detecor_Selector(Detector):
    def __init__(self, detector_name, config_path):
        self.detector = self._detector_selection(detector_name, config_path)

    def _detector_selection(self, detector_name, config_path):
        detector_map = {'mobilenet_ssd' : Mobilenet_Ssd}

        return detector_map[detector_name](config_path)

    def detect_image(self, image_path):
        self.detector.detect_image(image_path)

    def detect_video(self, video_path):
        self.detector.detect_video(video_path)

    def detect_webcam(self):
        self.detector.detect_webcam()