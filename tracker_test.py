# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:23:31 2018

@author: Administrator
"""

from tracker.tracker_selector import Tracker_Selector

if __name__ == "__main__":
    track = Tracker_Selector(tracker_name='deep_sort', config_path='./trackers.cfg')
    track.start_tracking()