# -*- coding: utf-8 -*-

#GEIgen code adapted from the pretreatment code by Abnder - 2018/12/19

import os
from scipy import misc as scisc
import imageio
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse


from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='', type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default='./gei.log', type=str,
                    help='Log file path. Default: ./gei.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=1, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 1')
opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num

T_H = 64
T_W = 64


def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    print(str_log, end='')

######################################
## GEI GENERATION
def GEIGEN(seq_info, pid):
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    count_frame = 0

    frames = []
    for _frame_name in frame_list:
        frame_path = os.path.join(seq_path, _frame_name)
        img = cv2.imread(frame_path)[:, :, 0]
        frames.append(img)
    try:
        gei = np.zeros(frames[0].shape,np.uint8)
        for i in range(len(frames)):
            alpha = 1.0/len(frames)
            beta = 1.0
            gei = cv2.addWeighted(frames[i], alpha, gei, beta, 0.0)
        save_path = os.path.join(out_dir, frame_list[0])
        imageio.imsave(save_path, gei)
        log_print(pid, FINISH,
                  'Saved to %s.'
              % (out_dir))
    except:
        print(seq_path)


if __name__ == '__main__':
    pool = Pool(WORKERS)
    results = list()
    pid = 0

    print('Pretreatment Start.\n'
          'Input path: %s\n'
          'Output path: %s\n'
          'Log file: %s\n'
          'Worker num: %d' % (
              INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))

    id_list = os.listdir(INPUT_PATH)
    id_list.sort()
    # Walk the input path
    for _id in id_list:
        seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
        seq_type.sort()
        for _seq_type in seq_type:
            view = os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))
            view.sort()
            for _view in view:
                seq_info = [_id, _seq_type, _view]
                out_dir = os.path.join(OUTPUT_PATH, *seq_info)
                os.makedirs(out_dir)
                results.append(
                    pool.apply_async(
                        GEIGEN,
                        args=(seq_info, pid)))
                sleep(0.02)
                pid += 1

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except Exception as e:
                if type(e) == MP_TimeoutError:
                    unfinish += 1
                    continue
                else:
                    print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                          i, type(e))
                    raise e
    pool.join()
