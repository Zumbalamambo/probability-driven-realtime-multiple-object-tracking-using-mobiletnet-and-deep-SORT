# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:53:30 2018

@author: Administrator
"""
from __future__ import division, print_function, absolute_import

import time
import configparser
import cv2
import numpy as np

from .deep_sort.application_util import preprocessing
from .deep_sort.application_util import visualization
from .deep_sort.deep_sort import nn_matching
from .deep_sort.deep_sort.detection import Detection
from .deep_sort.deep_sort.tracker import Tracker

from .tracker_template import Tracker_Template
from .utils import mot_challenge_util

class Tracker_Deep_Sort(Tracker_Template):
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        self.sequence_dir = config.get('deep_sort', 'sequence_dir')
        self.detection_file = config.get('deep_sort', 'detection_file')
        self.is_output = config.get('deep_sort', 'is_output') == 'True'
        self.output_file = config.get('deep_sort', 'output_file')
        self.min_confidence = float(config.get('deep_sort', 'min_confidence'))
        self.nms_max_overlap = float(config.get('deep_sort', 'nms_max_overlap'))
        self.min_detection_height = float(config.get('deep_sort', 'min_detection_height'))
        self.max_cosine_distance = float(config.get('deep_sort', 'max_cosine_distance'))
        self.nn_budget = int(config.get('deep_sort', 'nn_budget'))
        self.display = config.get('deep_sort', 'display') == 'True'

    def _create_detections(self, detection_mat, frame_idx, min_height=0):
        """Create detections for given frame index from the raw detection matrix.

        Parameters
        ----------
        detection_mat : ndarray
            Matrix of detections. The first 10 columns of the detection matrix are
            in the standard MOTChallenge detection format. In the remaining columns
            store the feature vector associated with each detection.
        frame_idx : int
            The frame index.
        min_height : Optional[int]
            A minimum detection bounding box height. Detections that are smaller
            than this value are disregarded.

        Returns
        -------
        List[tracker.Detection]
            Returns detection responses at given frame index.
        """
        frame_indices = detection_mat[:, 0].astype(np.int)
        mask = frame_indices == frame_idx

        detection_list = []
        for row in detection_mat[mask]:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))
        return detection_list

    def start_tracking(self):
        """
        Run multi-target tracker on a particular sequence.
        """
        seq_info = mot_challenge_util.gather_sequence_info(self.sequence_dir, self.detection_file)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget)
        tracker = Tracker(metric)
        results = []

        def frame_callback(vis, frame_idx):
            start_time = time.time()
            print("Processing frame %05d" % frame_idx)

            # Load image and generate detections.
            detections = self._create_detections(
                seq_info["detections"], frame_idx, self.min_detection_height)
            detections = [d for d in detections if d.confidence >= self.min_confidence]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Update tracker.
            tracker.predict()
            tracker.update(detections)

            # Update visualization.
            if self.display:
                image = cv2.imread(
                    seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
                vis.set_image(image.copy())
                vis.draw_detections(detections)
                vis.draw_trackers(tracker.tracks)
                vis.show_fps(0, 25, 'FPS: ' + str(round(1.0 / (time.time() - start_time), 1)))

            # Store results.
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                results.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        # Run tracker.
        if self.display:
            visualizer = visualization.Visualization(seq_info, update_ms=5)
        else:
            visualizer = visualization.NoVisualization(seq_info)

        visualizer.run(frame_callback)

        if self.is_output is True:
            # Store results.
            f = open(self.output_file, 'w')
            for row in results:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                    row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
