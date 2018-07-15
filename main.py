#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np

from utils.bounding_box_transform import to_tlwh

from detector.detector import Detector
from tracker.tracker import Tracker_temp

warnings.filterwarnings('ignore')
DETECT_FREQUENCY = 6
DOWN_SAMPLE_RATIO = 0.5
IS_DETECTION_DISPLAY = False
IS_TRACKING_DISPLAY = True

def tracking_by_detection(detector_name, tracker_name, videostream):
    det = Detector(detector_name, config_path='./detectors.cfg')
    tra = Tracker_temp(tracker_name, config_path='./trackers.cfg')
    output_file = './_output/' + video_stream[video_stream.rfind('/')+1:video_stream.rfind('.')] + '.txt'

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
                bbox = track.to_tlwh()
                result_list.append([frame_index, track.track_id, bbox[1], bbox[0], bbox[2], bbox[3]])
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
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('Average FPS:', round(counter / (time.time() - total_time), 1))
    print('Total eplased:', round(time.time() - total_time, 2))

    try:
        with open(output_file, 'w') as f:
            for result in result_list:
                #bounding_box = to_tlwh(np.array([result[2], result[3], result[4], result[5]]))
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (int(result[0]), int(result[1]), result[2], result[3], result[4], result[5]),file=f)
    except Exception as e:
        print(e)
        print('Something went wrong when writing output file!')
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detector_name = 'mobilenetv2_ssdlite' 
    tracker_name = 'deep_sort'
    video_stream = './_samples/MOT16-09.mp4'

    tracking_by_detection(detector_name, tracker_name, video_stream)