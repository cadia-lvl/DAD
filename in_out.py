import librosa
import json
import soundfile as sf
import numpy as np

def load_sample(path:str):
    #wf = wave.open(path, 'rb')
    #sr, nchannels = wf.getparams().framerate, wf.getparams().nchannels
    y, sr = librosa.core.load(path, sr=None, mono=True)
    return y, sr


def save_sample(y: np.ndarray, out_path: str, sr: int):
    '''
    '''
    sf.write(out_path, y, sr)

def dump_json(item, path: str):
    with open(path, 'w', encoding='utf-8') as json_f:
            json.dump(item, json_f, ensure_ascii=False, indent=4)

def duration(y: np.ndarray, sr=None):
    return librosa.core.get_duration(y, sr=sr)

def bool_input(text):
    cont = None
    while cont not in ['y', 'n']:
        cont = input(f"{text} Continue? [y/n]: ")
    return cont