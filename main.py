# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time

class Detector(object):
    def __init__(self, model_path='./_saved_models/MobileNetSSD_deploy.caffemodel',
                 prototxt='./_saved_models/MobileNetSSD_deploy.prototxt.txt', conf_threshold=0.4):

        self.det = {'model_path' : model_path,
                    'prototxt_path' : prototxt,
                    'conf_threshold' : conf_threshold
        }

        self.net = self._load_model(self.det['model_path'], self.det['prototxt_path'])
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat",
                           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                           "dog", "horse", "motorbike", "pottedplant", "sheep",
                           "sofa", "train", "tvmonitor"])
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def _load_model(self, model_path, prototxt_path):
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return net

    def detect_video(self, video_path=''):
        video_type = 'video'
        if(video_path == '' or video_path == 'camera'):
            video_type = 'camera'

        if(video_type == 'camera'):
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)

        total_counter = 0
        counter = 0
        averate_start_time = time.time()
        start_time = time.time()
        fps = 0
        while True:

            ret, frame = cap.read()
            frame = cv2.resize(frame, (300,300))

            (height, width ) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

            self.net.setInput(blob)
            detect_result = self.net.forward()

            for i in range(detect_result.shape[2]):
                conf = detect_result[0, 0, i, 2]
                if(conf > self.det['conf_threshold']):
                    idx = int(detect_result[0, 0, i, 1])

                    if(self.CLASSES[idx] in self.IGNORE):
                        continue

                    box = detect_result[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(self.CLASSES[idx], conf * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), self.COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

            counter += 1
            total_counter += 1

            if(counter % 5 == 0):
                fps = counter / (time.time() - start_time)
                counter = 0
                start_time = time.time()
            cv2.putText(frame, str(round(fps, 1)), org=(0, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1, color=(255,0,0), thickness=2)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) >= 0:  # Break with ESC
                break

        print('averate_FPS: ', total_counter / (time.time() - averate_start_time))
        # do a bit of cleanup
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
   det = Detector()
   #det.detect_video('_samples/MOT17-09-FRCNN.mp4')
   #det.detect_video('_samples/motor_bike.mp4')
   det.detect_video('camera')