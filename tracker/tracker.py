# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:46:22 2018

@author: Administrator
"""
from .tracker_template import Tracker_Template
from .tracker_deep_sort import Tracker_Deep_Sort

class Tracker_temp(Tracker_Template):
    def __init__(self, tracker_name, config_path):
        self.tracker = self._tracker_selection(tracker_name, config_path)

    def _tracker_selection(self, tracker_name, config_path):
        tracker_map = {'deep_sort' : Tracker_Deep_Sort}

        return tracker_map[tracker_name](config_path)

    def start_tracking(self, frame, boxes, scores):
        return self.tracker.start_tracking(frame, boxes, scores)

    def need_detection(self):
        return self.tracker.need_detection()

    def set_need_detection(self, value):
        self.tracker.set_need_detection(value)

    def is_detection_needed(self):
        return self.tracker.is_detection_needed()

    def set_detecion_needed(self, value):
        self.tracker.set_detecion_needed(value)