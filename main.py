import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import librosa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def check_directory(directory, out_path: str, checks: list, max_workers: int = 16):
    '''
    Runs check_sample on all samples in a directory
    Input arguments:
    * directory (str): A path to a directory containing one or more waveform files
    * out_path (str): A target path for an output file containing results
    * checks (list): A list of callable checks that return False if
    the sample passes the check
    * max_workers (int=16): The number of parallel workers
    '''
    executor = ProcessPoolExecutor(max_workers=max_workers)

    read_futures = []
    check_futures = []

    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)

        read_futures.append([
            fname, executor.submit(partial(load_sample, path))])

    samples = [(future[0], future[1].result()) for future in tqdm(read_futures)]

    for sample in samples:
        check_futures.append([sample[0], executor.submit(partial(
            check_sample, sample[1], checks))])

    answers = [(future[0], future[1].result()) for future in tqdm(check_futures)]

    with open(out_path, 'w') as o_f:
        for answer in answers:
            if answer[1]:
                o_f.write(f'{answer[0]}\t{answer[1][1]}\n')


def check_sample(y:np.ndarray, checks: list):
    '''
    Returns True if sample passes all checks.
    Input arguments:
    * y (np.ndarray): A [n] shaped numpy array containing the signal
    * checks (list): A list of callable checks that return False if
    the sample passes the check
    '''
    for check in checks:
        if check(y):
            return True, check.__name__
    return False


def load_sample(path:str):
    #wf = wave.open(path, 'rb')
    #sr, nchannels = wf.getparams().framerate, wf.getparams().nchannels
    y, _ = librosa.core.load(path, sr=None, mono=True)
    return y


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


def avg_db(y:np.ndarray, top_db: int=45):
    # TODO: Not finished and not currently used
    '''
    Return the average power level in dB. The signal is split into
    segments that we believe are non-silent to avoid accounting for
    silence periods in calculations

    * y (np.ndarray): A [n] shaped numpy array containing the signal
    * top_db (int): Threshold in decibels. Anything above threshold is
    assumed to be non-silence
    '''
    intervals = librosa.effects.split(y, top_db=top_db)
    total = 0.0
    num_frames = 0
    for splits in intervals:
        period = y[splits[0]:splits[1]]
        total += np.sum(librosa.core.amplitude_to_db(period))
        num_frames += splits[1] - splits[0]
    t1 = total/num_frames
    t2 = np.average(librosa.core.amplitude_to_db(y))
    t3 = np.average(librosa.core.amplitude_to_db(librosa.effects.trim(y, top_db=45)[0]))
    return t1, t2, t3

if __name__ == '__main__':
    check_directory('/home/atli/Data/lobe_data/margret_bad_small/', 'check_test.txt', checks=[signal_is_too_high, signal_is_too_low])
