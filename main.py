import os
import shutil
import errno
import json

import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm


def join_collections(a_dir: str, b_dir: str, out_dir: str):
    '''
    Joins two dataset exports into a single export. The export
    at a_dir will be used as the base for the joined exporrt which
    will be located at out_dir

    Input arguments:
    * a_dir (str): The path to the first dataset export
    * b_dir (str): The path to the second dataset export
    * out_dir (str): The target path for the joined export
    '''

    # Copy all at a_dir over to out_dir
    print(f"Copying data from {a_dir} to {out_dir}, this could take some time.")
    copy_tree(a_dir, out_dir)
    print(f"Done, adding data from {b_dir} to {out_dir}.")

    out_meta = json.load(open(os.path.join(a_dir, 'meta.json')))
    out_info = json.load(open(os.path.join(a_dir, 'info.json')))
    b_meta = json.load(open(os.path.join(b_dir, 'meta.json')))
    b_info = json.load(open(os.path.join(b_dir, 'info.json')))


    print(f'Adding audio from {b_dir}')
    for speaker in b_meta['speakers']:
        print(f'Copying audio from {speaker["name"]} in the second collection')
        # make sure that the speaker directory exists in the combined collection
        speaker_out_dir = os.path.join(out_dir, 'audio', str(speaker['id']))
        if not os.path.exists(speaker_out_dir):
            os.makedirs(speaker_out_dir)
        speaker_b_dir = os.path.join(b_dir, 'audio', str(speaker['id']))
        for fname in tqdm(os.listdir(os.path.join(speaker_b_dir))):
            shutil.copyfile(os.path.join(speaker_b_dir, fname),
                os.path.join(speaker_out_dir, fname))

    print(f'Adding text from {b_dir}')
    for fname in tqdm(os.listdir(os.path.join(b_dir, 'text'))):
        shutil.copyfile(os.path.join(b_dir, 'text', fname),
            os.path.join(out_dir, 'text', fname))

    print(f'Adding other information from {b_dir}')
    out_info = {**out_info, **b_info}
    for speaker in b_meta['speakers']:
        if speaker['id'] not in [s['id'] for s in out_meta['speakers']]:
            out_meta['speakers'] += speaker
    out_meta['sub_collection'] = b_meta['collection']

    with open(os.path.join(out_dir, 'info.json'), 'w', encoding='utf-8') as info_f:
        json.dump(out_info, info_f, ensure_ascii=False, indent=4)
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as meta_f:
        json.dump(out_meta, meta_f, ensure_ascii=False, indent=4)
    with open(os.path.join(out_dir, 'index.tsv'), 'a') as out_index_f, open(os.path.join(b_dir, 'index.tsv')) as b_index_f:
        for line in b_index_f:
            out_index_f.write(line)

def copy_tree(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:
        if exc.errno  == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise