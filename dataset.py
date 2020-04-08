import os
import json

from tqdm import tqdm

class Dataset:
    def __init__(self, path:str):
        '''
        Input arguments:
        * path (str): Path to the root dataset directory
        '''
        meta = json.load(open(os.path.join(path, 'meta.json')))
        self.meta = meta['collection']
        self.sub_meta = None
        if 'sub_collection' in meta:
            self.sub_meta = meta['sub_collection']
        self.speakers = [Speaker(**s) for s in meta['speakers']]

        # store both sentences and recordings in dicts where their
        # respective ids form the keys of the dictionaries
        self._sentences = {}
        self._recordings = {}

        info = json.load(open(os.path.join(path, 'info.json')))
        for rec_id, info in tqdm(info.items()):
            if info['text_info']['id'] not in self._sentences:
                self.add_sentence(info['text_info'],
                    info['other']['text_marked_bad'])
            self.add_recording(rec_id, info['recording_info'],
                info['other']['recording_marked_bad'])
            self.get_sentence(info['text_info']['id']).add_recording_id(rec_id)

    def add_sentence(self, info: dict, bad: bool):
        sentence = Sentence(**info)
        sentence.set_bad(bad)
        self._sentences[info['id']] = sentence

    def add_recording(self, id: int, info: dict, bad: bool):
        recording = Recording(**{'id':id, **info})
        recording.set_bad(bad)
        self._recordings[id] = recording

    def get_sentence(self, id):
        return self._sentences[id]

    def get_recording(self, id):
        return self._recordings[id]

    def num_recordings(self):
        return len(self._recordings)

    def num_sentences(self):
        return len(self._sentences)

class Sentence:
    def __init__(self, id: int, fname: str, score: float, text: str, pron: str):
        self.id = id
        self.fname = fname
        self.score = score
        self.text = text
        self.pron = pron

        self.recording_ids = set()
        self.bad = False

    def set_bad(self, is_bad: bool):
        self.bad = is_bad

    def add_recording_id(self, recording_id):
        self.recording_ids.add(recording_id)


class Recording:
    def __init__(self, id: int, recording_fname: str, sr: int, num_channels: int,
        bit_depth: int, duration: float):
        self.id = id
        self.fname = recording_fname
        self.sr = sr
        self.num_channels = num_channels
        self.bit_depth = bit_depth
        self.duration = duration
        self.bad = False

    def set_bad(self, is_bad: bool):
        self.bad = is_bad

class Speaker:
    def __init__(self, id: int, name: str, email: str, sex:str = '',
        age: int = 0, dialect: str = ''):
        self.id = id
        self.name = name
        self.email = email
        self.sex = sex
        self.age = age
        self.dialect = dialect



