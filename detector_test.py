from detector.detector_selector import Detecor_Selector

if __name__ == '__main__':
    det = Detecor_Selector(detector_name='mobilenet_ssd', config_path='detectors.cfg')

    # detect_image_test
    #det.detect_image('./_samples/MOT17-09-FRCNN/img1/000001.jpg')

    # detect_video_test
    det.detect_video('./_samples/MOT17-09-FRCNN.mp4')

    # detect_webcam_test
    #det.detect_webcam()
    
    
    
    #det = Detecor_Selector(detector_name='mobilenet', config_path='detectors.cfg')
    #det.detect_webcam()