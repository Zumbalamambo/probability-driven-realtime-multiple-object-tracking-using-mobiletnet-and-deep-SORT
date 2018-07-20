#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import errno
import sys
from timeit import time
import warnings
import cv2
import numpy as np

from utils.bounding_box_transform import to_tlwh

from detector.detector import Detector
from tracker.tracker import Tracker_temp

warnings.filterwarnings('ignore')
IS_DETECTION_DISPLAY = False
IS_TRACKING_DISPLAY = True

def open_with_mkdir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(os.path.dirname(path)):
            pass
        else:
            raise
    
    return open(path, 'w')

def tracking_by_detection(detector_name, tracker_name, video_stream, output_file, config_path='./config.cfg', show_image=True, detect_freq=1, down_sample_ratio=1.0, is_probability_driven_detect=True):
    det = Detector(detector_name, config_path)
    tra = Tracker_temp(tracker_name, config_path)
    video_capture = cv2.VideoCapture(video_stream)
    fps = 0.0
    step_counter = 0
    counter = 0
    first_time_flag = True
    start_time = time.time()
    total_time = time.time()
    result_list = []
    frame_index = 1
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        (h, w) = frame.shape[:2]
        frame_resized = cv2.resize(frame, (int(w * down_sample_ratio), int(h * down_sample_ratio)))
        if((step_counter % detect_freq == 0) or counter == 0 or (is_probability_driven_detect == True and tra.is_detection_needed() == True)):
            results = det.detect_image_frame(frame_resized, to_xywh=True)
            boxes = np.array([result[1:5] for result in results])
            scores = np.array([result[5] for result in results])
            tra.set_detecion_needed(False)

        tracker, detections = tra.start_tracking(frame_resized, boxes, scores)
        # Call the tracker
        if(IS_TRACKING_DISPLAY is True):
            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update > 1 :
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0] / down_sample_ratio), int(bbox[1] / down_sample_ratio)), (int(bbox[2] / down_sample_ratio), int(bbox[3] / down_sample_ratio)),(255,255,255), 2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0] / down_sample_ratio), int(bbox[1] / down_sample_ratio)),0, 5e-3 * 200, (0,255,0),2)
                bbox = track.to_tlwh()
                result_list.append([frame_index, track.track_id, bbox[0] / down_sample_ratio, bbox[1] / down_sample_ratio, bbox[2] / down_sample_ratio, bbox[3] / down_sample_ratio])

        if(IS_DETECTION_DISPLAY is True):
            for detection in detections:
                bbox = detection.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0] / down_sample_ratio), int(bbox[1] / down_sample_ratio)), (int(bbox[2] / down_sample_ratio), int(bbox[3] / down_sample_ratio)),(255,0,0), 2)

        counter += 1
        step_counter += 1
        if(step_counter % detect_freq == 0):
            fps  = step_counter / (time.time()- start_time)
            print(fps)
            step_counter = 0
            cv2.putText(frame, 'FPS:' + str(round(fps, 1)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2)
            start_time = time.time()
            if(first_time_flag is True):
                step_counter = 0
                counter = 0
                total_time = time.time()
                first_time_flag = False

        if(show_image == True):
            cv2.imshow('image', frame)
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps = counter / (time.time() - total_time)
    print('Average FPS:', round(fps, 1))
    print('Total eplased:', round(time.time() - total_time, 2))

    try:
        f = open_with_mkdir(output_file)
        for result in result_list:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (int(result[0]), int(result[1]), float(result[2]), float(result[3]), float(result[4]), float(result[5])),file=f)
        f.close()
    except Exception as e:
        print(e)
        print('Something went wrong when writing output file!')

    
    video_capture.release()
    cv2.destroyAllWindows()
    try:
        del det
        del tra
        del tracker
        del detections
        del result_list
        del video_capture
        del frame
        del frame_resized
        del results
        del boxes
        del bbox
        del scores
        del step_counter
        del first_time_flag
        del start_time
        del total_time
        del frame_index
    except Exception as e:
        print(e)
    return fps, counter

if __name__ == '__main__':
    detector_name = 'yolo' 
    tracker_name = 'deep_sort'
    video_stream = './_samples/MOT16-09.mp4'
    # video_stream = 'D:/_videos/MOT2016/train/MOT16-09/img1/%06d.jpg'
    output_file = 'MOT16-09'

    tracking_by_detection(detector_name, tracker_name, video_stream, output_file)