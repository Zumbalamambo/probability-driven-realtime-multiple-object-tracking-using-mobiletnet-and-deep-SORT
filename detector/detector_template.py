# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:28:15 2018

@author: Administrator
"""

class Detector_Template(object):
    def __init__(self):
        pass

    def _image_preprocessing(self, image_frame):
        pass

    def _image_postprocessing(self, image_frame):
        pass

    def detect_image(self, image_path):
        pass

    def detect_video(self, video_path):
        pass

    def detect_webcam(self):
        pass