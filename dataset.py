import os
import sox
import json
import operator
import shutil
from random import shuffle
from collections import defaultdict
from check import check, signal_is_too_high, signal_is_too_low
from in_out import load_sample, save_sample, dump_json, bool_input, duration
from tqdm import tqdm
from trim import trim_sample
import librosa




class Dataset:
    def __init__(self, path:str):
        '''
        Input arguments:
        * path (str): Path to the root dataset directory
        '''
        self.path = path
        self.name = os.path.basename(path)
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

        print("Building from info.json ...")
        self.info = json.load(open(os.path.join(path, 'info.json')))
        for rec_id, info in tqdm(self.info.items()):
            rec_id = int(rec_id)
            if info['text_info']['id'] not in self._sentences:
                self.add_sentence(parse_sentence(info['text_info'],
                    info['other']['text_marked_bad'], path))
            self.add_recording(parse_recording(rec_id, info['recording_info'],
                info['other']['recording_marked_bad'], info['collection_info']['user_id'],
                info['collection_info']['user_id'], info['text_info']['id'], path))
            self.get_sentence(info['text_info']['id']).add_recording_id(rec_id)

    @property
    def num_recordings(self):
        return len(self._recordings)

    @property
    def num_sentences(self):
        return len(self._sentences)

    @property
    def sentence_objs(self):
        return self._sentences.values()

    @property
    def recording_objs(self):
        return self._recordings.values()

    def add_sentence(self, sentence):
        self._sentences[sentence.id] = sentence

    def add_recording(self, recording):
        self._recordings[recording.id] = recording

    def get_sentence(self, id):
        return self._sentences[id]

    def get_recording(self, id):
        return self._recordings[id]

    def get_multi_sentences(self, thresh: int = 1):
        '''
        Return a list of Sentence objects that have more recordings
        then a specific threshold

        Input arguments (thresh:int=1): The threshold
        '''
        multis = []
        for s_id, s in self._sentences.items():
            if s.num_recordings > thresh:
                multis.append(s)
        return multis

    def get_duration(self, format: str = 'seconds'):
        '''
        Returns the total duration of the dataset in the given format
        with precision of two decimal places

        Input arguments:
        * format (str='seconds'): Can be 'seconds', 'minutes', 'hours'
        '''
        formats = {'seconds': 1, 'minutes': 60, 'hours': 3600}
        assert format in formats, "format not valid"
        return round(sum(r.duration for r in self.recording_objs)/formats[format],2)

    def trim_recordings(self, top_db: float = 45):
        print("Iterating recordings ...")
        for recording in tqdm(self.recording_objs):
            self.trim_recording(recording.id, save_info=False)
        self.save_info()

    def trim_recording(self, id:int, top_db: float = 45,
        save_info: bool = True):
        self.get_recording(id).trim(top_db)
        self.update_recording_info(id)
        if save_info:
            self.save_info()

    def delete_bad_recordings(self, checks: list = [signal_is_too_high, signal_is_too_low]):
        print("Iterating recordings ... ")
        bad_ids = set()
        for recording in tqdm(self.recording_objs):
            is_bad = self.check_recording(recording.id, checks)
            if is_bad:
                bad_ids.add(recording.id)
        for rec_id in bad_ids:
            self.delete_recording(rec_id, save=False)

        self.save_index()
        self.save_info()
        print(f"Deleted {len(bad_ids)} bad recordings.")

    def check_recording(self, id:int, checks: list):
        # TODO: replace with a Recording method
        return self.get_recording(id).check(checks)

    def verify_recording(self, id:int, checks: list):
        '''
        Returns True if the recording passes all the checks
        '''
        # TODO: Replace with a Recording method
        is_bad, _ = self.check_sample(id, checks)
        return not is_bad

    def show_recording_report(self):
        '''
        Get information about sample rate, bit depth and number
        of channels across all recordings
        '''
        srs, bds, ncs = defaultdict(int), defaultdict(int), defaultdict(int)
        print('Iterating files...')
        for rec in tqdm(self.recording_objs):
            srs[str(rec.sr)] += 1
            bds[str(rec.bit_depth)] += 1
            ncs[str(rec.num_channels)] += 1
        report(srs, bds, ncs)

    def show_sox_report(self):
        srs, bds, ncs = defaultdict(int), defaultdict(int), defaultdict(int)
        print('Iterating files...')
        for rec in tqdm(self.recording_objs):
            srs[str(rec.sox_sample_rate)] += 1
            bds[str(rec.sox_bit_depth)] += 1
            ncs[str(rec.sox_num_channels)] += 1
        report(srs, bds, ncs)

    def delete_recording(self, id: int, save: bool = True):
        '''
        Delete a recording from a dataset.
        Input arguments:
        * id (int): The id of the recording
        * save (bool=False): If True, save info.json and index.tsv
        '''
        # remove from OS
        self.get_recording(id).os_delete()
        sentence = self.get_sentence(self.get_recording(id).sentence_id)
        sentence.remove_recording_id(id)
        del self._recordings[id]
        del self.info[str(id)]
        if save:
            self.save_index()
            self.save_info()

    def convert(self, sr: int = 16000, bit_depth: int = 16,
        n_channels: int = 1, name: str = '', overwrite: bool = False):
        '''
        Convert a dataset to the given format. NOTE: This will write
        over existing files!

        Input arguments:
        * sr (int=16000): The desired sample rate
        * bit_depth (int=16): The desired bit depth
        * n_channels (int=1): The desired number of channels
        * name (str=''): The name of the converted dataset
        * overwrite (bool=False): If True, replace /audio and info.json
        with the converted versions
        '''
        tfm = sox.Transformer()
        tfm.convert(samplerate=sr, bitdepth=bit_depth, n_channels=n_channels)

        # create converted audio file structure
        for speaker in self.speakers:
            os.makedirs(os.path.join(self.path,
                f'audio_{sr}_{bit_depth}_{n_channels}', str(speaker.id)))

        print('converting files...')
        for recording in tqdm(self.recording_objs):
            recording.convert(sr, bit_depth, n_channels,
            transformer=tfm)
            self.update_recording_info(recording.id)
        self.save_info(fname=f'info_{sr}_{bit_depth}_{n_channels}.json')
        if overwrite:
            # delete /audio/* and info.json and replace with new data
            shutil.rmtree(os.path.join(self.path, 'audio'))
            os.rename(os.path.join(self.path, f'audio_{sr}_{bit_depth}_{n_channels}'),
                os.path.join(self.path, 'audio'))
            os.remove(os.path.join(self.path, 'info.json'))
            os.rename(os.path.join(self.path, f'info_{sr}_{bit_depth}_{n_channels}.json'),
                os.path.join(self.path, 'info.json'))

    def update_recording_info(self, id:int, info: dict = {}):
        '''
        Updates the current information for a recording with a given
        id to self.info. Note this does not change info.json on disk.
        Call self.save_info() for that.
        '''
        if not info:
            info = self.get_recording(id).info
        self.info[str(id)]['recording_info'] = info

    def save_index(self, sort_by: str = ''):
        '''
        Create a new index.tsv file for this collection given a
        sort critera

        Input arguments:
        * sort_by (str=''): The criteria used to sort the dataset. Available
        options:
            - 'score': Sorts the sentences by coverage score
            - 'random'
            - '': The current order
        '''
        if sort_by == 'random':
            sorted_sentences = shuffle(self.sentence_objs)
        elif sort_by == 'score':
            sorted_sentences = sorted(self.sentence_objs,
                key=operator.attrgetter('score'), reverse=True)
        else:
            sorted_sentences = list(self.sentence_objs)
        with open(os.path.join(self.path, 'index.tsv'), 'w') as index_f:
            for sentence in sorted_sentences:
                for rec_id in sentence.recording_ids:
                    rec = self.get_recording(rec_id)
                    index_f.write(f'{rec.user_id}\t{rec.fname}\t{sentence.fname}\n')

    def save_info(self, fname: str = ''):
        '''
        If recordings or sentences have been removed, this method can be
        called to update info.json
        '''
        if fname == '':
            fname = 'info.json'
        dump_json(self.info, os.path.join(self.path, fname))

    def create_subset(self, num_samples: int, sort_by: str = 'same',
        out_path: str = '', exclude_bad: bool = True, max_hours: float = 0.0,
        overwrite: bool = False):
        '''
        Create a subset of this dataset by taking the first <num_samples> when
        sorted by a certain criteria. The new dataset is stored at <out_path> if
        specified, else in the same directory as the current dataset. Note that
        this method will pick the first recording for each sentence if the sentence
        has multiple recordings

        Input arguments:
        * num_samples (str):
        * sort_by (str='same'): The criteria used to sort the dataset. Available
        options:
            - 'score': Sorts the sentences by coverage score
            - 'random'
            - 'same': The current order
        * out_path (str=''): The target root directory of the new dataset. If it
        is = '' then it will be stored at ../<name>_{num_samples}_{sort_by}
        * exclude_bad (bool=True): If True, no recordings that have been marked as
        bad will be included.
        * overwrite (bool=False): If True, it will overwrite any directory that may
        exists on <out_path>
        * TODO : max_hours (float=0.0): If specified, we add sentence up to a duration
        limit rather than sample limit

        '''
        if out_path == '':
            out_path = os.path.join(self.path, '..', f'{self.name}_{num_samples}_{sort_by}')
        if os.path.exists(out_path) and overwrite:
            shutil.rmtree(out_path)
        # create new directories
        audio_dir = os.path.join(out_path, 'audio')
        for speaker in self.speakers:
            os.makedirs(os.path.join(audio_dir, str(speaker.id)))
        text_dir = os.path.join(out_path, 'text')
        os.makedirs(text_dir)

        if sort_by == 'random':
            sorted_sentences = shuffle(self.sentence_objs)
        elif sort_by == 'score':
            sorted_sentences = sorted(self.sentence_objs,
                key=operator.attrgetter('score'), reverse=True)
        else:
            sorted_sentences = list(self.sentence_objs)
        print('Exporting new dataset ...')
        # copy audio and text files and create new info.json
        info = {}
        for sentence in sorted_sentences[:num_samples]:
            sentence.copy_to(text_dir)
            for r_id in sentence.recording_ids:
                rec = self.get_recording(r_id)
                rec.copy_to(os.path.join(audio_dir, str(rec.user_id)))
                info[rec.id] = self.info[str(rec.id)]
        dump_json(info, os.path.join(out_path, 'info.json'))

        # create a new meta.json
        meta = json.load(open(os.path.join(self.path, 'meta.json')))
        meta['subset'] = {
            'original_dir': self.path,
            'num_samples': num_samples,
            'sort_by': sort_by,
            'exclude_bad': exclude_bad}

        dump_json(meta, os.path.join(out_path, 'meta.json'))

        # create a new index.txt
        ds = Dataset(out_path)
        ds.save_index(sort_by)

    def export(self, out_path: str, format: str = 'basic',
        sort_by: str = 'same', overwrite: bool = False):
        '''
        Create an export of this dataset in various formats
        Input arguments:
        * out_path (str): Where the export should be stored
        * sort_by (str='same'): The criteria used to sort the dataset. Available
        options:
            - 'score': Sorts the sentences by coverage score
            - 'random'
            - 'same': The current order
        * overwrite (bool=False): If True, replace anything existing
        at <out_path> with the export
        TODO: Actually use and and more formats
        * format (str): The format of the export. Supports:
            - 'basic':  out_path/
                            audio/
                                0001.wav
                                ...
                            text/
                                0001.txt
                                ...
        '''

        audio_dir = os.path.join(out_path, 'audio')
        text_dir = os.path.join(out_path, 'text')
        if os.path.exists(out_path) and overwrite:
            shutil.rmtree(out_path)
        os.makedirs(audio_dir)
        os.makedirs(text_dir)

        if sort_by == 'random':
            sorted_sentences = shuffle(self.sentence_objs)
        elif sort_by == 'score':
            sorted_sentences = sorted(self.sentence_objs,
                key=operator.attrgetter('score'), reverse=True)
        else:
            sorted_sentences = list(self.sentence_objs)

        ind = 1
        fill = len(str(self.num_recordings))
        for sentence in sorted_sentences:
            for r_id in sentence.recording_ids:
                recording = self.get_recording(r_id)
                shutil.copyfile(sentence.path,
                    os.path.join(text_dir, f'{str(ind).zfill(fill)}.txt'))
                shutil.copyfile(recording.path,
                    os.path.join(audio_dir, f'{str(ind).zfill(fill)}.wav'))
                ind += 1


class Sentence:
    def __init__(self, id: int, fname: str, score: float, text: str,
        pron: str, collection_path : str):
        self.id = id
        self.fname = fname
        self.score = score
        self.text = text
        self.pron = pron
        self.collection_path = collection_path

        self.recording_ids = set()
        self.bad = False

    def set_bad(self, is_bad: bool):
        self.bad = is_bad

    def add_recording_id(self, recording_id: int):
        self.recording_ids.add(recording_id)

    def remove_recording_id(self, recording_id: int):
        self.recording_ids.remove(recording_id)

    def copy_to(self, dir):
        '''
        Copies the file that corresponds to this sentence
        to <dir> with the same filename
        '''
        shutil.copyfile(self.path, os.path.join(dir, self.fname))

    @property
    def path(self):
        return os.path.join(self.collection_path, 'text', self.fname)

    @property
    def num_recordings(self):
        return len(self.recording_ids)

    @property
    def info(self):
        '''
        Returns the current sentence information in the same format
        as info.json
        '''
        return {
            'id': self.id,
            'fname': self.fname,
            'score': self.score,
            'text': self.text,
            'pron': self.pron}


class Recording:
    def __init__(self, id: int, recording_fname: str, sr: int, num_channels: int,
        bit_depth: int, duration: float, user_id: int, session_id: int, sentence_id: int,
        collection_path: str):

        self.id = id
        self.sentence_id = sentence_id
        self.fname = recording_fname
        self.sr = sr
        self.num_channels = num_channels
        self.bit_depth = bit_depth
        self.duration = duration
        self.bad = False
        self.collection_path = collection_path
        self.user_id = user_id
        self.session_id = session_id

    def set_bad(self, is_bad: bool):
        self.bad = is_bad

    def copy_to(self, dir):
        '''
        Copies the file that corresponds to this recording
        to <dir> with the same filename
        '''
        shutil.copyfile(self.path, os.path.join(dir, self.fname))

    def os_delete(self):
        os.remove(self.path)

    @property
    def sox_sample_rate(self):
        return sox.file_info.sample_rate(self.path)

    @property
    def sox_num_channels(self):
        return sox.file_info.channels(self.path)

    @property
    def sox_bit_depth(self):
        return sox.file_info.bitrate(self.path)

    @property
    def path(self):
        return os.path.join(self.collection_path, 'audio', str(self.user_id),
            self.fname)

    def convert(self, sr: int, bit_depth: int, n_channels: int, out_dir: str = '',
        transformer = None):
        '''
        Convert a recording to a given format
        Input arguments:
        * sr (int): The desired sample rate
        * bit_depth (int): The desired bit depth
        * n_channels (int): The desired number of channels
        * out_dir (str): The directory to save the converted recording
        * transformer (sox.Transformer / None): A Sox Transformer instance
        '''
        if out_dir == '':
            out_path = os.path.join(self.collection_path,
                f'audio_{sr}_{bit_depth}_{n_channels}', str(self.user_id), self.fname)
        else:
            out_path = os.path.join(out_dir, self.fname)
        if transformer is None:
            tfm = sox.Transformer()
            tfm.convert(samplerate=sr, bitdepth=bit_depth, n_channels=n_channels)
        transformer.build(self.path, out_path)
        self.sr = sr
        self.bit_depth = bit_depth
        self.num_channels = n_channels

    def trim(self, top_db: float = 45):
        y, sr = load_sample(self.path)
        trimmed = trim_sample(y, top_db=top_db)
        self.duration = duration(trimmed, sr=sr)
        save_sample(trimmed, self.path, sr)

    def check(self, checks: list):
        is_bad = check(self.path, checks)
        return is_bad

    @property
    def info(self):
        '''
        Returns the current recording info in the same format as info.json
        '''
        return {
            'recording_fname': self.fname,
            'sr': self.sr,
            'num_channels': self.num_channels,
            'bit_depth': self.bit_depth,
            'duration': self.duration}

class Speaker:
    def __init__(self, id: int, name: str, email: str, sex:str = '',
        age: int = 0, dialect: str = ''):
        self.id = id
        self.name = name
        self.email = email
        self.sex = sex
        self.age = age
        self.dialect = dialect

def parse_sentence(info: dict, bad: bool, collection_path: str):
    sentence = Sentence(**{'collection_path': collection_path, **info})
    sentence.set_bad(bad)
    return sentence

def parse_recording(id: int, info: dict, bad: bool, user_id: int,
    session_id: int, sentence_id: int, collection_path: str):
    recording = Recording(**{'id':id, 'user_id': user_id,
        'session_id': session_id, 'sentence_id': sentence_id,
        'collection_path': collection_path, **info})
    recording.set_bad(bad)
    return recording

def report(srs, bds, ncs):
    print(f'{"-"*30}\nREPORT \n{"-"*30}\n\nSample Rate \n{"-"*30}')
    for s, num in srs.items():
        print(f'{num} recordings have {s}')
    print(f'\nBit Depth \n{"-"*30}')
    for b, num in bds.items():
        print(f'{num} recordings have {b}')
    print(f'\nNum Channels \n{"-"*30}')
    for n, num in ncs.items():
        print(f'{num} recordings have {n}')
    print(f'{"-"*30}')

if __name__ == '__main__':
    ds = Dataset('/home/atli/Data/test_data/7')
    ds.export('/home/atli/Data/test_data/7_simple')
    #ds.convert(overwrite=True)
    #ds.trim_recordings()
    #print(ds.get_duration(format='hours'))