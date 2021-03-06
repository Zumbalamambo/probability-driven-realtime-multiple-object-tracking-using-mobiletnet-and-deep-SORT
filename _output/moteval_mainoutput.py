"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

# python moteval.py --groundtruths D:/_videos/MOT2016/train/ --tests ./res/MOT16/data/

import argparse
import glob
import os
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in 

Milan, Anton, et al. 
"Mot16: A benchmark for multi-object tracking." 
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str, help='Directory containing ground truth files.')   
    parser.add_argument('--tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logging.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

if __name__ == '__main__':
    loglevel = 'info'
    fmt = 'mot15-2D'

    groundtruths = 'D:/_videos/2DMOT2015/train/'
    combine_list = ['./main_output/']

    print(len(combine_list))

    loglevel = getattr(logging, loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(loglevel))        
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    for combine in combine_list:
        gtfiles = glob.glob(os.path.join(groundtruths, '*/gt/gt.txt'))
        tsfiles = [f for f in glob.glob(os.path.join(combine, '*.txt')) if not os.path.basename(f).startswith('eval') and os.path.basename(f).find('_result.txt') == -1]

        logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
        logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
        logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
        logging.info('Loading files.')
        
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=fmt, min_confidence=1)) for f in gtfiles])
        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=fmt)) for f in tsfiles])    

        mh = mm.metrics.create()    
        accs, names = compare_dataframes(gt, ts)
        
        logging.info('Running metrics')
        
        summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
        logging.info('Completed')

        with open('./results_mainoutput.txt', 'a+') as f:
                f.write('---------------------------------------------------------------------------------------------')
                f.write(str(combine))
                f.write('\r\n')
                f.write(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
                f.write('---------------------------------------------------------------------------------------------')
                f.write('\r\n')