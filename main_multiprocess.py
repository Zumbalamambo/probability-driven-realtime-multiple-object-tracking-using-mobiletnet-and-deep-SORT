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

detect_frequency = 4
down_sampling_ratio = 0.4
is_detection_display = True
is_tracking_display = True


def main():
    manager = Manager()
    lock = Lock()
    d = manager.dict()
    
    video_capture = cv2.VideoCapture('./_samples/MOT17-09-FRCNN.mp4')
    ret, frame = video_capture.read()
    
    #video_capture = cv2.VideoCapture(0)

    tra = Tracker_temp(tracker_name='deep_sort', config_path='./trackers.cfg')
    
    p = Process(target=detection, args=(d, lock))
    p.start()

    fps = 0.0
    d['frame_counter'] = 0
    index = 1
    d['detector_ready'] = False
    d['tracker_ready'] = False
    frame_counter = 0
    step_counter = 0
    counter = 0
    first_time_flag = True
    start_time = time.time()
    total_time = time.time()
    freq = 1
    next_frame_flag = True
    while True:
        if(next_frame_flag is True):
            ret, frame= video_capture.read()
            next_frame_flag = False
            if ret != True:
                break
            (h, w) = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * down_sampling_ratio), int(h * down_sampling_ratio)))
            d['%d_frame'%d['frame_counter']] = frame
            d['tracker_ready'] = True

        if(d['detector_ready'] == True):
            try:
                if(index == 1):
                    tracker_results, detection_results = tra.start_tracking(frame, d['%d_boxes'%(0)], d['%d_scores'%(0)])
                else:
                    tracker_results, detection_results = tra.start_tracking(frame, d['%d_boxes'%index], d['%d_scores'%index])
            except Exception as e:
                print(d.keys())
                print(e)
                break

            if((d['frame_counter'] % detect_frequency == 0) and (frame_counter != 0)):
                d['detector_ready'] = False
                index += detect_frequency
            next_frame_flag = True
            # Call the tracker
            if(is_tracking_display is True):
                for track in tracker_results.tracks:
                    if track.is_confirmed() and track.time_since_update >1 :
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            if(is_detection_display is True):
                for detection_result in detection_results:
                    bbox = detection_result.to_tlbr()
                    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            counter += 1
            step_counter += 1
            frame_counter += 1
            if(step_counter % detect_frequency == 0):
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

            d['frame_counter'] += 1
            cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    print('Average FPS:', round(counter / (time.time() - total_time), 1))
    print('Total eplased:', round(time.time() - total_time, 2))
    video_capture.release()
    cv2.destroyAllWindows()

def detection(d, lock):
    det = Detector(detector_name='mobilenetv2_ssdlite', config_path='./detectors.cfg')
    freq = 1
    counter = 0
    while True:
        if(d['tracker_ready'] == True):
            results = det.detect_image_frame(d['%d_frame'%d['frame_counter']], to_xywh=True)
            d['%d_boxes'%d['frame_counter']] = np.array([result[1:5] for result in results])
            d['%d_scores'%d['frame_counter']] = np.array([result[5] for result in results])
            d['detector_ready'] = True
            d['tracker_ready'] = False

if __name__ == '__main__':
    main()
