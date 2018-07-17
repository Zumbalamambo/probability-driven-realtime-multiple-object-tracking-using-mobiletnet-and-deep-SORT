from main import tracking_by_detection
from detector.detector import Detector
from tracker.tracker import Tracker_temp
from utils import generate_detections
import configparser
import sys
import os
sys.path.append('./tracker/deep_sort')
from tracker.deep_sort import deep_sort_app

MODE_MAP = {
    'train' : '/train/',
    'test' : '/test/',
    'all' : ['/train/', '/test/']
}
MOT16_train_seq = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
MOT16_test_seq = ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14']

if __name__ == '__main__':
    MODE = 'train'
    MODE_DIR = '/train/'
    MOT_DIR = 'D:/_videos/MOT2016'
    output_dir = './_output/'
    video_stream_list = []
    seq_name_list = []

    config = configparser.ConfigParser()
    config.read('config.cfg')
    model_filename = config.get('deep_sort', 'model_path')

    if MODE == 'train':
        video_stream_list = [MODE_MAP['train'] + MOT16_train for MOT16_train in MOT16_train_seq]
        seq_name_list = MOT16_train_seq
    elif MODE == 'test':
        video_stream_list = [MODE_MAP['test'] + MOT16_test for MOT16_test in MOT16_test_seq]
        seq_name_list = MOT16_test_seq
    elif MODE == 'all':
        video_stream_list = [MODE_MAP['train'] + MOT16_train for MOT16_train in MOT16_train_seq] + [MODE_MAP['test'] + MOT16_test for MOT16_test in MOT16_test_seq]
        seq_name_list = MOT16_train_seq + MOT16_test_seq
    else:
        raise NotImplementedError


    det = Detector(detector_name='yolo', config_path='config.cfg')
    tra = Tracker_temp(tracker_name='deep_sort', config_path='config.cfg')

    for i, video_stream in enumerate(video_stream_list):
        video_stream = MOT_DIR + video_stream + '/img1/%06d.jpg'
        det.generate_detecions(video_stream=video_stream, output_file=output_dir+seq_name_list[i])

    encoder = generate_detections.create_box_encoder(model_filename, batch_size=32)
    output_dir_list = [output_dir + seq_name + '_det.txt' for seq_name in seq_name_list]
    generate_detections.generate_detections(encoder, MOT_DIR + MODE_DIR, output_dir, output_dir_list)

    sequences = os.listdir(MOT_DIR + MODE_DIR)
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(MOT_DIR + MODE_DIR, sequence)
        detection_file = os.path.join(output_dir, "%s.npy" % sequence)
        output_file = os.path.join(output_dir, "%s.txt" % sequence)
        deep_sort_app.run(
            sequence_dir, detection_file, output_file, 0.0, 1.0, 0, 0.2, 100, display=True)

