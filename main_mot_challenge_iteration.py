from main import tracking_by_detection

MODE_MAP = {
    'train' : '/train/',
    'test' : '/test/',
    'all' : ['/train/', '/test/']
}
MOT16_train_seq = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
MOT16_test_seq = ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14']
MOT15_train_seq = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
MOT15_test_seq = ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher', 'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']

if __name__ == '__main__':
    MODE = 'MOT16train'
    MOT_DIR = 'D:/_videos/MOT2016'
    detector_name = 'yolo' 
    tracker_name = 'deep_sort'
    video_stream_list = []
    seq_name_list = []

    # MOT2016
    if MODE == 'MOT16train':
        video_stream_list = [MODE_MAP['train'] + MOT16_train for MOT16_train in MOT16_train_seq]
        seq_name_list = MOT16_train_seq
    elif MODE == 'MOT16test':
        video_stream_list = [MODE_MAP['test'] + MOT16_test for MOT16_test in MOT16_test_seq]
        seq_name_list = MOT16_test_seq
    elif MODE == 'MOT16all':
        video_stream_list = [MODE_MAP['train'] + MOT16_train for MOT16_train in MOT16_train_seq] + [MODE_MAP['test'] + MOT16_test for MOT16_test in MOT16_test_seq]
        seq_name_list = MOT16_train_seq + MOT16_test_seq
    # MOT2015
    elif MODE == 'MOT15train':
        video_stream_list = [MODE_MAP['train'] + MOT15_train for MOT15_train in MOT15_train_seq]
        seq_name_list = MOT16_train_seq
    elif MODE == 'MOT15test':
        video_stream_list = [MODE_MAP['test'] + MOT15_test for MOT15_test in MOT15_test_seq]
        seq_name_list = MOT16_test_seq
    elif MODE == 'MOT15all':
        video_stream_list = [MODE_MAP['train'] + MOT15_train for MOT15_train in MOT15_train_seq] + [MODE_MAP['test'] + MOT15_test for MOT15_test in MOT15_test_seq]
        seq_name_list = MOT15_train_seq + MOT15_test_seq
    else:
        raise NotImplementedError

    fps_list = []
    nb_frames_list = []
    for i, video_stream in enumerate(video_stream_list):
        video_stream = MOT_DIR + video_stream + '/img1/%06d.jpg'
        fps, nb_frames = tracking_by_detection(detector_name, tracker_name, video_stream=video_stream, output_file=seq_name_list[i], show_image=False)
        fps_list.append(fps)
        nb_frames_list.append(nb_frames)
    
    print(fps_list, nb_frames_list, sum([fps * nb_frames for fps, nb_frames in zip(fps_list, nb_frames_list)]) / sum(nb_frames_list))

    """ # Version 2
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
    """