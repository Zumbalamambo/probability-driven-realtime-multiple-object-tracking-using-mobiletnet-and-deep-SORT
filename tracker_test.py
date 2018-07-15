# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:23:31 2018

@author: Administrator
"""

from tracker.tracker import Tracker_temp

if __name__ == "__main__":
    tracker = Tracker_temp(tracker_name='deep_sort', config_path='./config.cfg')
    tracker.start_tracking()