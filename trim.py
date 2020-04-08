import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from io import load_sample, save_sample

import librosa
from tqdm import tqdm


def trim_sample(y:np.ndarray, top_db: float = 45):
    return librosa.effects.trim(y, top_db=top_db)[0]


def trim_directory(directory, top_db: float = 45, max_workers: int = 16):
    '''
    Runs trim_sample on all samples in a directory
    Input arguments:
    * directory (str): A path to a directory containing one or more waveform files
    * top_db (float): The threshold at both ends of recordings that has to be
    crossed to be counted as non-silence
    * max_workers (int=16): The number of parallel workers
    '''
    executor = ProcessPoolExecutor(max_workers=max_workers)
    batch_sz = 1000

    read_futures = []
    trim_futures = []
    save_futures = []

    # Load up to 1000 samples at a time
    paths = [os.path.join(directory, fname) for fname in os.listdir(directory)]
    for p in tqdm(paths):
        y, sr = load_sample(p)
        trimmed = trim_sample(y, top_db)
        save_sample(trimmed, p, sr)
