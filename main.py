import os

import librosa
import matplotlib.pyplot as plt
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm

def check_directory(directory, out_path: str, max_workers: int = 16):

    executor = ProcessPoolExecutor(max_workers=max_workers)

    read_futures = []
    high_futures = []

    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)

        read_futures.append([
            fname, executor.submit(partial(load_sample, path))])

    samples = [(future[0], future[1].result()) for future in tqdm(read_futures)]

    for sample in samples:
        high_futures.append([sample[0], executor.submit(partial(
            signal_is_too_high, sample[1]))])

    answers = [(future[0], future[1].result()) for future in tqdm(high_futures)]

    with open(out_path, 'w') as o_f:
        for answer in answers:
            if answer[1]:
                o_f.write(f'{answer[0]}\n')

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