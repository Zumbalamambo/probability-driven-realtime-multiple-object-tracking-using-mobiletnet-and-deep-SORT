# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:46:22 2018

@author: Administrator
"""
from .tracker_template import Tracker_Template
from .tracker_deep_sort import Tracker_Deep_Sort

class Tracker_Selector(Tracker_Template):
    def __init__(self, tracker_name, config_path):
        self.tracker = self._tracker_selection(tracker_name, config_path)

    def _tracker_selection(self, tracker_name, config_path):
        tracker_map = {'deep_sort' : Tracker_Deep_Sort}

        return tracker_map[tracker_name](config_path)

    def start_tracking(self):
        self.tracker.start_tracking()
