import librosa
import soundfile as sf

def load_sample(path:str):
    #wf = wave.open(path, 'rb')
    #sr, nchannels = wf.getparams().framerate, wf.getparams().nchannels
    y, sr = librosa.core.load(path, sr=None, mono=True)
    return y, sr


def save_sample(y: np.ndarray, out_path: str, sr: int):
    '''
    '''
    sf.write(out_path, y, sr)