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

detect_frequency = 4
down_sampling_ratio = 0.4
is_detection_display = False
is_tracking_display = False

if __name__ == '__main__':
    det = Detector(detector_name='mobilenetv2_ssdlite', config_path='./detectors.cfg')
    tra = Tracker_temp(tracker_name='deep_sort', config_path='./trackers.cfg')

    video_capture = cv2.VideoCapture('./_samples/MOT17-09-FRCNN.mp4')
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
        #frame = cv2.resize(frame, (int(w * down_sampling_ratio), int(h * down_sampling_ratio)))
        if(step_counter % detect_frequency == 0 or counter == 0):
            results = det.detect_image_frame(frame, to_xywh=True)
            boxes = np.array([result[1:5] for result in results])
            scores = np.array([result[5] for result in results])

        tracker, detections = tra.start_tracking(frame, boxes, scores)
        # Call the tracker
        if(is_tracking_display is True):
            for track in tracker.tracks:
                #if track.is_confirmed() and track.time_since_update >1 :
                #    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        if(is_detection_display is True):
            for detection in detections:
                bbox = detection.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        counter += 1
        step_counter += 1
        if(step_counter % detect_frequency == 0):
            fps  = step_counter / (time.time()- start_time)
            print(fps)
            step_counter = 0
            cv2.putText(frame, 'FPS:' + str(round(fps, 1)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2)
            start_time = time.time()
            if(first_time_flag is True):
                counter = 0
                first_time_flag = False

        #cv2.imshow('', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS:', round(counter / (time.time() - total_time), 1))
    print('Total eplased:', round(time.time() - total_time, 2))
    video_capture.release()
    cv2.destroyAllWindows()
"""'
# @TOBETEST
class video_writer():
    def __init__(self, width, height, output_file='output.avi', detection_output=True, detection_file='detection.txt'):
        self.videowriter = self._init_video_writer(width, height)
        self.frame_index = -1
        if(detection_output is True):
            self.det_file = open(detection_file, 'w+')

    def _init_video_writer(self, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        return cv2.VideoWriter('output.avi', fourcc, 15, (width, height))

    def write_frame(self, frame, detections):
        self.videowriter.write(frame)
        self.frame_index += 1
        self.det_file.write(str(self.frame_index)+' ')
        if len(detections) != 0:
            for i in range(0,len(detections)):
                self.det_file.write(str(detections[i][0]) + ' '+str(detections[i][1]) + ' '+str(detections[i][2]) + ' '+str(detections[i][3]) + ' ')
        self.det_file.write('\n')

    def release(self):
        self.videowriter.release()
        self.det_file.close()

if __name__ == '__main__':
    det = Detecor_Selector(detector_name='mobilenet_ssd', config_path='detectors.cfg')


   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

   # deep_sort
    model_filename = './_saved_models/resources/networks/mars-small128.pb'
    encoder = create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = False

    video_capture = cv2.VideoCapture('./_samples/MOT17-09-FRCNN.mp4')
    #video_capture = cv2.VideoCapture(0)

    if writeVideo_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        vw = video_writer(w, h)

    fps = 0.0
    step_counter = 0
    counter = 0
    start_time = time.time()
    total_time = time.time()
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break

        (h, w) = frame.shape[:2]
        #frame = cv2.resize(frame, (int(w/2), int(h/2)))
        if(counter == 0 or step_counter % 1 == 0):
            boxs = det.detect_image(frame, to_xywh=True)

            features = encoder(frame,boxs)

            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            #detections = [Detection(bbox, 1.0) for bbox in zip(boxs)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # Call the tracker
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        #for det in detections:
        #    bbox = det.to_tlbr()
        #    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)


        if writeVideo_flag:
            vw.write_frame(frame, boxes)

        counter += 1
        step_counter += 1
        if(counter == 0 or step_counter % 1 == 0):
            fps  = step_counter / (time.time()- start_time)
            step_counter = 0
            cv2.putText(frame, 'FPS:' + str(round(fps, 1)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) , 2)
            start_time = time.time()

        cv2.imshow('', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS:', round(counter / (time.time() - total_time), 1))
    print('Total eplased:', round(time.time() - total_time, 2))
    video_capture.release()
    if writeVideo_flag:
        vw.release()
    cv2.destroyAllWindows()
"""