#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from detector.detector import Detector
from tracker.tracker import Tracker_temp

from tracker.deep_sort.application_util import preprocessing
from tracker.deep_sort.deep_sort import nn_matching
from tracker.deep_sort.deep_sort.detection import Detection
from tracker.deep_sort.deep_sort.tracker import Tracker
from tracker.deep_sort.tools.generate_detections import generate_detections as gdet
from tracker.deep_sort.tools.generate_detections import create_box_encoder
warnings.filterwarnings('ignore')


DETECT_FREQUENCY = 6
DOWN_SAMPLE_RATIO = 0.5
IS_DETECTION_DISPLAY = False
IS_TRACKING_DISPLAY = True

if __name__ == '__main__':
    det = Detector(detector_name='mobilenetv2_ssdlite', config_path='./detectors.cfg')
    tra = Tracker_temp(tracker_name='deep_sort', config_path='./trackers.cfg')

    video_capture = cv2.VideoCapture('./_samples/MOT17-09.mp4')
    #video_capture = cv2.VideoCapture(0)
    fps = 0.0
    step_counter = 0
    counter = 0
    first_time_flag = True
    start_time = time.time()
    total_time = time.time()
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        (h, w) = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * DOWN_SAMPLE_RATIO), int(h * DOWN_SAMPLE_RATIO)))
        if((step_counter % DETECT_FREQUENCY == 0) or counter == 0 or (tra.is_detection_needed() == True)):
            results = det.detect_image_frame(frame, to_xywh=True)
            boxes = np.array([result[1:5] for result in results])
            scores = np.array([result[5] for result in results])
            tra.set_detecion_needed(False)

        tracker, detections = tra.start_tracking(frame, boxes, scores)
        # Call the tracker
        if(IS_TRACKING_DISPLAY is True):
            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update > 3 :
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        if(IS_DETECTION_DISPLAY is True):
            for detection in detections:
                bbox = detection.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        counter += 1
        step_counter += 1
        if(step_counter % DETECT_FREQUENCY == 0):
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

        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('Average FPS:', round(counter / (time.time() - total_time), 1))
    print('Total eplased:', round(time.time() - total_time, 2))
    video_capture.release()
    cv2.destroyAllWindows()
