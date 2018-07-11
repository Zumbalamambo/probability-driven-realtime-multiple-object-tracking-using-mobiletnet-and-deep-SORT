#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from multiprocessing import Process, Value, Array, Manager, Lock

from detector.detector import Detector
from tracker.tracker import Tracker_temp

from tracker.deep_sort.application_util import preprocessing
from tracker.deep_sort.deep_sort import nn_matching
from tracker.deep_sort.deep_sort.detection import Detection
from tracker.deep_sort.deep_sort.tracker import Tracker
from tracker.deep_sort.tools.generate_detections import generate_detections as gdet
from tracker.deep_sort.tools.generate_detections import create_box_encoder
warnings.filterwarnings('ignore')

def main():
    ODD_DETECT_FREQUENCY = 5
    EVEN_DETECT_FREQUENCY = 3
    DETECT_SKIP_STATUS = False
    DOWN_SAMPLE_RATIO = 0.5
    IS_DETECTION_DISPLAY = False
    IS_TRACKING_DISPLAY = True
    manager = Manager()
    d = manager.dict()
    
    video_capture = cv2.VideoCapture('./_samples/MOT17-09-FRCNN.mp4')
    ret, frame = video_capture.read()
    
    #video_capture = cv2.VideoCapture(0)

    tra = Tracker_temp(tracker_name='deep_sort', config_path='./trackers.cfg')
    
    p = Process(target=detection, args=(d,))
    p.start()

    fps = 0.0
    d['frame_counter'] = 0
    d['index'] = 1
    tracker_ready_enable = True
    d['detector_ready'] = False
    d['tracker_ready'] = False
    frame_counter = 0
    step_counter = 0
    counter = 0
    first_time_flag = True
    start_time = time.time()
    total_time = time.time()
    next_frame_flag = True
    while True:
        if(next_frame_flag is True):
            ret, frame= video_capture.read()
            next_frame_flag = False
            if ret != True:
                break
            (h, w) = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * DOWN_SAMPLE_RATIO), int(h * DOWN_SAMPLE_RATIO)))
            if(tracker_ready_enable is True):
                d['%d_frame'%d['index']] = frame
                tracker_ready_enable = False
                d['tracker_ready'] = True

        if(d['detector_ready'] == True):
            try:
                tracker_results, detection_results = tra.start_tracking(frame, d['%d_boxes'%d['index']], d['%d_scores'%d['index']])
            except Exception as e:
                print(d.keys())
                print(e)
                break

            if((step_counter % ODD_DETECT_FREQUENCY == 0 and DETECT_SKIP_STATUS == True) and (frame_counter != 0)):
                d['detector_ready'] = False
                tracker_ready_enable = True
                d['index'] += ODD_DETECT_FREQUENCY

            elif((step_counter % EVEN_DETECT_FREQUENCY == 0 and DETECT_SKIP_STATUS == False) and (frame_counter != 0)):
                d['detector_ready'] = False
                tracker_ready_enable = True
                d['index'] += EVEN_DETECT_FREQUENCY

            next_frame_flag = True
            # Call the tracker
            if(IS_TRACKING_DISPLAY is True):
                for track in tracker_results.tracks:
                    if track.is_confirmed() and track.time_since_update >1 :
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            if(IS_DETECTION_DISPLAY is True):
                for detection_result in detection_results:
                    bbox = detection_result.to_tlbr()
                    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            counter += 1
            step_counter += 1
            if((step_counter % ODD_DETECT_FREQUENCY == 0 and DETECT_SKIP_STATUS == True) or (step_counter % EVEN_DETECT_FREQUENCY == 0 and DETECT_SKIP_STATUS ==  False) and frame_counter != 0):
                DETECT_SKIP_STATUS = not DETECT_SKIP_STATUS
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

            frame_counter += 1
            d['frame_counter'] += 1
            cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    print('Average FPS:', round(counter / (time.time() - total_time), 1))
    print('Total eplased:', round(time.time() - total_time, 2))
    video_capture.release()
    cv2.destroyAllWindows()

def detection(d):
    det = Detector(detector_name='mobilenet_ssd', config_path='./detectors.cfg')
    while True:
        if(d['tracker_ready'] == True):
            results = det.detect_image_frame(d['%d_frame'%d['index']], to_xywh=True)
            d['%d_boxes'%d['index']] = np.array([result[1:5] for result in results])
            d['%d_scores'%d['index']] = np.array([result[5] for result in results])
            d['tracker_ready'] = False
            d['detector_ready'] = True

if __name__ == '__main__':
    main()
