from detector.detector import Detector

if __name__ == '__main__':
    det = Detector(detector_name='mobilenet_ssd', config_path='config.cfg')

    # detect_image_test
    #det.detect_image('./_samples/MOT17-09-FRCNN/img1/000055.jpg')

    # detect_video_test
    det.detect_video('./_samples/MOT17-09-FRCNN.mp4')

    # detect_webcam_test
    #det.detect_webcam()



    #det = Detector(detector_name='mobilenet', config_path='detectors.cfg')
    #det.detect_webcam()