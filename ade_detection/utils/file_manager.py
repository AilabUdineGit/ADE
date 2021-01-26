import requests
import zipfile
import pickle 
import shutil
import json
import os
from os import path

from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
import ade_detection.utils.localizations as loc
import sys 


def rmdir(dir_path):
    shutil.rmtree(dir_path)


def wget(url, path):
    file = requests.get(url)
    open(path, 'wb').write(file.content)


def wget_with_progressbar(url, path):

    '''see https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads'''

    with open(path, 'wb') as f:
        LOG.info('Downloading %s' % path)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write('\r[%s%s]' % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()


def decompress_zip(zip_path):
    '''Decompress a zip archive in tmp/'''

    if not path.isfile(zip_path):
        raise FileNotFoundError(zip_path + ' not found')

    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(loc.TMP_PATH)


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f,  protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_json(obj, path):
    with open(path, 'w') as outfile:
        json.dump(obj, outfile)


def json_stringify(obj):
    return json.dumps(obj, indent=4)


def from_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def to_txt(text, path):
    with open(path, 'w') as file:
        file.write(text)


def from_txt(path):
    with open(path, 'r') as file:
        return file.read()


def to_id(array, path):
    with open(path, 'w') as file:
        file.write('\n'.join([str(x) for x in array]))


def from_id(path):
    with open(path, 'r') as file:
        return file.read().split('\n')