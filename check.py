import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from io import load_sample

import librosa
import numpy as np
from tqdm import tqdm


def check_directory(directory, good_path: str, bad_path:str, checks: list,
    max_workers: int = 16):
    '''
    Runs check_sample on all samples in a directory
    Input arguments:
    * directory (str): A path to a directory containing one or more
    waveform files
    * out_path (str): A target path for an output file containing results
    * checks (list): A list of callable checks that return False if
    the sample passes the check
    * max_workers (int=16): The number of parallel workers
    '''
    futures = []
    executor = ProcessPoolExecutor(max_workers=max_workers)

    with open(good_path, 'w') as gf, open(bad_path, 'w') as bf:
        for p in [os.path.join(directory, fname) for fname in os.listdir(directory)]:
            futures.append([p, executor.submit(partial(check, p, checks))])
        answers = [(future[0], future[1].result()) for future in tqdm(futures)]
        for answer in tqdm(answers):
            if answer[1][0]:
                bf.write(f"{p}\t{check}\n")
            else:
                gf.write(f"{p}\n")


def check(p, checks):
    y, sr = load_sample(p)
    return check_sample(y, checks)


def check_sample(y:np.ndarray, checks: list):
    '''
    Returns False if sample passes all checks.
    Input arguments:
    * y (np.ndarray): A [n] shaped numpy array containing the signal
    * checks (list): A list of callable checks that return False if
    the sample passes the check
    '''
    for check in checks:
        if check(y):
            return True, check.__name__
    return False, ""

def signal_is_too_high(y:np.ndarray, thresh: float = -4.5, num_frames :int = 1):
    '''
    If the signal exceeds the treshold for a certain number of frames or
    more consectuively, it is deemed too high
    Input arguments:
    * y (np.ndarray): A [n] shaped numpy array containing the signal
    * thresh (float=-4.5): A db threshold
    * num_frames (int=20): A number of frames
    '''
    db = librosa.amplitude_to_db(y)
    thresh_count = 0
    for i in range(len(db)):
        if db[i] > thresh:
            thresh_count += 1
            if thresh_count == num_frames:
                return True
        else:
            thresh_count = 0
    return False


def signal_is_too_low(y:np.ndarray, thresh: float = -15):
    '''
    If the signal never exceeds the treshold it is deemed too low
    Input arguments:
    * y (np.ndarray): A [n] shaped numpy array containing the signal
    * thresh (float=-18): A db threshold
    '''
    db = librosa.amplitude_to_db(y)
    return not any(db_val > thresh for db_val in db)