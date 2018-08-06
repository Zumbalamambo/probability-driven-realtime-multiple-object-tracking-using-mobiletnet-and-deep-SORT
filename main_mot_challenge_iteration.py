from main import tracking_by_detection
import sys

# def sizeof_fmt(num, suffix='B'):
#     ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
#     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
#         if abs(num) < 1024.0:
#             return "%3.1f%s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f%s%s" % (num, 'Yi', suffix)

# for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()), key= lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))

MODE_MAP = {
    'MOT16train' : '/train/',
    'MOT16test' : '/test/',
    'MOT15train' : '/train/',
    'MOT15test' : '/test/',
}
MOT16_train_seq = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
MOT16_test_seq = ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14']
MOT15_train_seq = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
MOT15_test_seq = ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher', 'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']

if __name__ == '__main__':
    MODE = 'MOT15train'
    #MOT_DIR = 'D:/_videos/MOT2016'
    MOT_DIR = 'D:/_videos/2DMOT2015'
    OUTPUT_DIR = './_output/'
    video_stream_list = []
    seq_name_list = []
    SINGLE_TASK = 'skip1_with_prob_yolov3_tiny'

    multi_task_map = {
        'yolov3':{
            'detector_name':'yolo',
            'tracker_name':'deep_sort', 
            'info':{
                # YOLOv3
                ## vanilla
                'vanilla_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/vanilla/', 'detect_frequency':1, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'vanilla_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/vanilla_down_sample/', 'detect_frequency':1, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                ## Skip 1
                'skip1_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip1/', 'detect_frequency':2, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip1_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip1_down_sample/', 'detect_frequency':2, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip1_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip1_with_prob/', 'detect_frequency':2, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip1_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip1_with_prob_down_sample/', 'detect_frequency':2, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 2
                'skip2_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip2/', 'detect_frequency':3, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip2_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip2_down_sample/', 'detect_frequency':3, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip2_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip2_with_prob/', 'detect_frequency':3, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip2_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip2_with_prob_down_sample/', 'detect_frequency':3, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 3
                'skip3_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip3/', 'detect_frequency':4, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip3_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip3_down_sample/', 'detect_frequency':4, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip3_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip3_with_prob/', 'detect_frequency':4, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip3_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip3_with_prob_down_sample/', 'detect_frequency':4, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 4
                'skip4_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip4/', 'detect_frequency':5, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip4_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip4_down_sample/', 'detect_frequency':5, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip4_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip4_with_prob/', 'detect_frequency':5, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip4_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip4_with_prob_down_sample/', 'detect_frequency':5, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 5
                'skip5_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip5/', 'detect_frequency':6, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip5_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip5_down_sample/', 'detect_frequency':6, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip5_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip5_with_prob/', 'detect_frequency':6, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip5_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip5_with_prob_down_sample/', 'detect_frequency':6, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 6
                'skip6_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip6/', 'detect_frequency':7, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip6_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip6_down_sample/', 'detect_frequency':7, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip6_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip6_with_prob/', 'detect_frequency':7, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip6_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip6_with_prob_down_sample/', 'detect_frequency':7, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 7
                'skip7_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip7/', 'detect_frequency':8, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip7_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip7_down_sample/', 'detect_frequency':8, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip7_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip7_with_prob/', 'detect_frequency':8, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip7_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip7_with_prob_down_sample/', 'detect_frequency':8, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 8
                'skip8_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip8/', 'detect_frequency':9, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip8_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip8_down_sample/', 'detect_frequency':9, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip8_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip8_with_prob/', 'detect_frequency':9, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip8_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip8_with_prob_down_sample/', 'detect_frequency':9, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 9
                'skip9_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip9/', 'detect_frequency':10, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip9_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip9_down_sample/', 'detect_frequency':10, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip9_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip9_with_prob/', 'detect_frequency':10, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip9_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip9_with_prob_down_sample/', 'detect_frequency':10, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 10
                'skip10_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip10/', 'detect_frequency':11, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip10_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip10_down_sample/', 'detect_frequency':11, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip10_with_prob_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip10_with_prob/', 'detect_frequency':11, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip10_with_prob_down_sample_yolov3_tiny': {'output_dir':OUTPUT_DIR + 'yolov3_tiny/skip10_with_prob_down_sample/', 'detect_frequency':11, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
            },
        },
        'mobilenet ssd':{
            'detector_name':'mobilenet_ssd',
            'tracker_name':'deep_sort',
            'info':{
                # Mobilnet SSD
                ## vanilla
                'vanilla_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/vanilla/', 'detect_frequency':1, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'vanilla_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/vanilla_down_sample/', 'detect_frequency':1, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                ## Skip 1
                'skip1_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip1/', 'detect_frequency':2, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip1_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip1_down_sample/', 'detect_frequency':2, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip1_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip1_with_prob/', 'detect_frequency':2, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip1_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip1_with_prob_down_sample/', 'detect_frequency':2, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 2
                'skip2_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip2/', 'detect_frequency':3, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip2_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip2_down_sample/', 'detect_frequency':3, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip2_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip2_with_prob/', 'detect_frequency':3, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip2_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip2_with_prob_down_sample/', 'detect_frequency':3, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 3
                'skip3_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip3/', 'detect_frequency':4, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip3_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip3_down_sample/', 'detect_frequency':4, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip3_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip3_with_prob/', 'detect_frequency':4, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip3_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip3_with_prob_down_sample/', 'detect_frequency':4, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 4
                'skip4_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip4/', 'detect_frequency':5, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip4_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip4_down_sample/', 'detect_frequency':5, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip4_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip4_with_prob/', 'detect_frequency':5, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip4_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip4_with_prob_down_sample/', 'detect_frequency':5, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 5
                'skip5_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip5/', 'detect_frequency':6, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip5_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip5_down_sample/', 'detect_frequency':6, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip5_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip5_with_prob/', 'detect_frequency':6, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip5_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip5_with_prob_down_sample/', 'detect_frequency':6, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 6
                'skip6_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip6/', 'detect_frequency':7, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip6_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip6_down_sample/', 'detect_frequency':7, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip6_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip6_with_prob/', 'detect_frequency':7, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip6_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip6_with_prob_down_sample/', 'detect_frequency':7, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 7
                'skip7_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip7/', 'detect_frequency':8, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip7_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip7_down_sample/', 'detect_frequency':8, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip7_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip7_with_prob/', 'detect_frequency':8, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip7_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip7_with_prob_down_sample/', 'detect_frequency':8, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 8
                'skip8_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip8/', 'detect_frequency':9, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip8_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip8_down_sample/', 'detect_frequency':9, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip8_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip8_with_prob/', 'detect_frequency':9, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip8_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip8_with_prob_down_sample/', 'detect_frequency':9, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 8
                'skip8_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip8/', 'detect_frequency':9, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip8_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip8_down_sample/', 'detect_frequency':9, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip8_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip8_with_prob/', 'detect_frequency':9, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip8_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip8_with_prob_down_sample/', 'detect_frequency':9, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 9
                'skip9_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip9/', 'detect_frequency':10, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip9_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip9_down_sample/', 'detect_frequency':10, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip9_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip9_with_prob/', 'detect_frequency':10, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip9_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip9_with_prob_down_sample/', 'detect_frequency':10, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 10
                'skip10_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip10/', 'detect_frequency':11, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip10_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip10_down_sample/', 'detect_frequency':11, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip10_with_prob_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip10_with_prob/', 'detect_frequency':11, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip10_with_prob_down_sample_mobilenetssd': {'output_dir':OUTPUT_DIR + 'mobilenetssd/skip10_with_prob_down_sample/', 'detect_frequency':11, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
            }
        },
        # 'mobilenetv2 SSDlite':{
        #     # Mobilnetv2 SSDlite
        #     ## vanilla
        #     'vanilla_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/vanilla/', 'detect_frequency':1, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
        #     'vanilla_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/vanilla_down_sample/', 'detect_frequency':1, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
        #     ## Skip 1
        #     'skip1_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip1/', 'detect_frequency':2, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
        #     'skip1_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip1_down_sample/', 'detect_frequency':2, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
        #     'skip1_with_prob_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip1_with_prob/', 'detect_frequency':2, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
        #     'skip1_with_prob_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip1_with_prob_down_sample/', 'detect_frequency':2, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
        #     ## Skip 2
        #     'skip2_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip2/', 'detect_frequency':3, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
        #     'skip2_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip2_down_sample/', 'detect_frequency':3, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
        #     'skip2_with_prob_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip2_with_prob/', 'detect_frequency':3, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
        #     'skip2_with_prob_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip2_with_prob_down_sample/', 'detect_frequency':3, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
        #     ## Skip 4
        #     'skip4_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip4/', 'detect_frequency':5, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
        #     'skip4_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip4_down_sample/', 'detect_frequency':5, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
        #     'skip4_with_prob_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip4_with_prob/', 'detect_frequency':5, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
        #     'skip4_with_prob_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip4_with_prob_down_sample/', 'detect_frequency':5, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
        #     ## Skip 6
        #     'skip6_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip6/', 'detect_frequency':7, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
        #     'skip6_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip6_down_sample/', 'detect_frequency':7, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
        #     'skip6_with_prob_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip6_with_prob/', 'detect_frequency':7, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
        #     'skip6_with_prob_down_sample_mobilenetv2_ssdlite': {'detector_name':'mobilenetv2_ssdlite', 'tracker_name':'deep_sort', 'output_dir':OUTPUT_DIR + 'mobilenetv2_ssdlite/skip6_with_prob_down_sample/', 'detect_frequency':7, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
        # }
        'squeezenetv1_0':{
            'detector_name':'squeezenetv1_0',
            'tracker_name':'deep_sort',
            'info':{
                # Squeezenetv1.0
                ## vanilla
                'vanilla_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/vanilla/', 'detect_frequency':1, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'vanilla_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/vanilla_down_sample/', 'detect_frequency':1, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                ## Skip 1
                'skip1_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip1/', 'detect_frequency':2, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip1_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip1_down_sample/', 'detect_frequency':2, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip1_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip1_with_prob/', 'detect_frequency':2, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip1_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip1_with_prob_down_sample/', 'detect_frequency':2, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 2
                'skip2_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip2/', 'detect_frequency':3, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip2_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip2_down_sample/', 'detect_frequency':3, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip2_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip2_with_prob/', 'detect_frequency':3, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip2_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip2_with_prob_down_sample/', 'detect_frequency':3, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 3
                'skip3_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip3/', 'detect_frequency':4, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip3_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip3_down_sample/', 'detect_frequency':4, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip3_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip3_with_prob/', 'detect_frequency':4, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip3_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip3_with_prob_down_sample/', 'detect_frequency':4, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 4
                'skip4_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip4/', 'detect_frequency':5, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip4_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip4_down_sample/', 'detect_frequency':5, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip4_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip4_with_prob/', 'detect_frequency':5, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip4_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip4_with_prob_down_sample/', 'detect_frequency':5, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 5
                'skip5_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip5/', 'detect_frequency':6, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip5_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip5_down_sample/', 'detect_frequency':6, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip5_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip5_with_prob/', 'detect_frequency':6, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip5_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip5_with_prob_down_sample/', 'detect_frequency':6, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 6
                'skip6_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip6/', 'detect_frequency':7, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip6_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip6_down_sample/', 'detect_frequency':7, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip6_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip6_with_prob/', 'detect_frequency':7, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip6_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip6_with_prob_down_sample/', 'detect_frequency':7, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 7
                'skip7_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip7/', 'detect_frequency':8, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip7_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip7_down_sample/', 'detect_frequency':8, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip7_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip7_with_prob/', 'detect_frequency':8, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip7_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip7_with_prob_down_sample/', 'detect_frequency':8, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 8
                'skip8_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip8/', 'detect_frequency':9, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip8_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip8_down_sample/', 'detect_frequency':9, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip8_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip8_with_prob/', 'detect_frequency':9, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip8_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip8_with_prob_down_sample/', 'detect_frequency':9, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 9
                'skip9_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip9/', 'detect_frequency':10, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip9_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip9_down_sample/', 'detect_frequency':10, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip9_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip9_with_prob/', 'detect_frequency':10, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip9_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip9_with_prob_down_sample/', 'detect_frequency':10, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
                ## Skip 10
                'skip10_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip10/', 'detect_frequency':11, 'down_sample_ratio':1.0, 'is_probability_driven_detect': False},
                'skip10_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip10_down_sample/', 'detect_frequency':11, 'down_sample_ratio':0.5, 'is_probability_driven_detect': False},
                'skip10_with_prob_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip10_with_prob/', 'detect_frequency':11, 'down_sample_ratio':1.0, 'is_probability_driven_detect': True},
                'skip10_with_prob_down_sample_squeeze_v1_0': {'output_dir':OUTPUT_DIR + 'squeeze_v1_0/skip10_with_prob_down_sample/', 'detect_frequency':11, 'down_sample_ratio':0.5, 'is_probability_driven_detect': True},
            }
        }
    }

    # MOT2016
    if MODE == 'MOT16train':
        video_stream_list = [MODE_MAP['MOT16train'] + MOT16_train for MOT16_train in MOT16_train_seq]
        seq_name_list = MOT16_train_seq
    elif MODE == 'MOT16test':
        video_stream_list = [MODE_MAP['MOT16test'] + MOT16_test for MOT16_test in MOT16_test_seq]
        seq_name_list = MOT16_test_seq
    elif MODE == 'MOT16all':
        video_stream_list = [MODE_MAP['MOT16train'] + MOT16_train for MOT16_train in MOT16_train_seq] + [MODE_MAP['MOT16test'] + MOT16_test for MOT16_test in MOT16_test_seq]
        seq_name_list = MOT16_train_seq + MOT16_test_seq
    # MOT2015
    elif MODE == 'MOT15train':
        video_stream_list = [MODE_MAP['MOT15train'] + MOT15_train for MOT15_train in MOT15_train_seq]
        seq_name_list = MOT15_train_seq
    elif MODE == 'MOT15test':
        video_stream_list = [MODE_MAP['MOT15test'] + MOT15_test for MOT15_test in MOT15_test_seq]
        seq_name_list = MOT15_test_seq
    elif MODE == 'MOT15all':
        video_stream_list = [MODE_MAP['MOT15train'] + MOT15_train for MOT15_train in MOT15_train_seq] + [MODE_MAP['MOT15test'] + MOT15_test for MOT15_test in MOT15_test_seq]
        seq_name_list = MOT15_train_seq + MOT15_test_seq
    else:
        raise NotImplementedError


    for task, task_info in multi_task_map.items():
        print(task, task_info)
        detector_name = task_info['detector_name']
        tracker_name = task_info['tracker_name']
        tra_by_det = tracking_by_detection(detector_name, tracker_name)

        for info, info_dict in task_info['info'].items():
            print(info, info_dict)
            print('info', info)
            if(info.find('prob') == -1):
                continue
                
            fps_list = []
            nb_frames_list = []
            for i, video_stream in enumerate(video_stream_list):
                print('%d / %d'%(i+1, len(video_stream_list)))
                video_stream = MOT_DIR + video_stream + '/img1/%06d.jpg'
                fps, nb_frames = tra_by_det.tracking_by_detection(video_stream=video_stream, output_file=info_dict['output_dir']+seq_name_list[i]+'.txt', show_image=False, detect_freq=info_dict['detect_frequency'], down_sample_ratio=info_dict['down_sample_ratio'], is_probability_driven_detect=info_dict['is_probability_driven_detect'])
                fps_list.append(fps)
                nb_frames_list.append(nb_frames)

            with open('fps.txt', 'a+') as f:
                print('info', info)
                f.write(str(info))
                f.write('\r\n')
                f.write(str(fps_list))
                f.write('\r\n')
                f.write(str(nb_frames_list))
                f.write('\r\n')
                f.write(str(sum([fps * nb_frames for fps, nb_frames in zip(fps_list, nb_frames_list)]) / sum(nb_frames_list)))
                f.write('\r\n')
                f.write('--------------------------------------------------------------------\r\n')

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