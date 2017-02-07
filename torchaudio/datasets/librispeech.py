"""LibriSpeech Datasets"""
from __future__ import print_function

import fnmatch
import glob
import os

from torchaudio import data


def _make_dataset(path):
    utterances = []
    transcript_files = _get_files(path, pattern='*.txt')
    for tfile in transcript_files:
        with open(tfile) as file:
            lines = file.readlines()
        for line in lines:
            filestr, transcript = line.split(' ', 1)
            try:
                flac_file = _librispeech_flac_filename(path, filestr)
            except IndexError:
                print('filestr of unexpected formatting: {}'.format(filestr))
                print('error in {}'.format(tfile))
                continue
            utterances.append((flac_file, transcript.strip()))

    return utterances


def _get_files(directory, pattern, recursive=True):
    """ Return the full path to all files in directory matching the
    specified pattern.
    pattern should be a glob style pattern (e.g. "*.wav")
    """

    # This yields an iterator which really speeds up looking through large,
    # flat directories
    if recursive is False:
        matches = glob.iglob(os.path.join(directory, pattern))
        return matches

    # If we want to recurse, use os.walk instead
    # pylint: disable=R0204
    matches = list()
    for root, _, filenames in os.walk(directory):
        # pylint: disable=W0640
        matches.extend(map(lambda ss: os.path.join(root, ss),
                           fnmatch.filter(filenames, pattern)))

    return matches


def _librispeech_flac_filename(root, filestr):
    parts = filestr.split('-')
    return os.path.join(root, parts[0], parts[1], '{}.flac'.format(filestr))


# pylint: disable=C0111,R0903
class LibriSpeech(data.TarDataset, data.TranscriptionDataset):

    url = None
    dirname = None
    filename = None

    def __init__(self, root, **kwargs):
        path = self.download_or_untar(root)
        utts = _make_dataset(path)
        super().__init__(utts, **kwargs)


# pylint: disable=C0111,R0903
class LibriSpeechDevClean(LibriSpeech):

    url = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    dirname = 'LibriSpeech/dev-clean'
    filename = 'dev-clean.tar.gz'

class LibriSpeechDevOther(LibriSpeech):

    url = "http://www.openslr.org/resources/12/dev-other.tar.gz"
    dirname = 'LibriSpeech/dev-other'
    filename = 'dev-other.tar.gz'


# pylint: disable=C0111,R0903
class LibriSpeechTestClean(LibriSpeech):

    url = "http://www.openslr.org/resources/12/test-clean.tar.gz"
    dirname = 'LibriSpeech/test-clean'
    filename = 'test-clean.tar.gz'


# pylint: disable=C0111,R0903
class LibriSpeechTestOther(LibriSpeech):

    url = "http://www.openslr.org/resources/12/test-other.tar.gz"
    dirname = 'LibriSpeech/test-other'
    filename = 'test-other.tar.gz'


# pylint: disable=C0111,R0903
class LibriSpeechTrainClean100(LibriSpeech):

    url = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    dirname = 'LibriSpeech/train-clean-100'
    filename = 'train-clean-100.tar.gz'


# pylint: disable=C0111,R0903
class LibriSpeechTrainClean360(LibriSpeech):

    url = "http://www.openslr.org/resources/12/train-clean-360.tar.gz"
    dirname = 'LibriSpeech/train-clean-360'
    filename = 'train-clean-360.tar.gz'

# pylint: disable=C0111,R0903
class LibriSpeechTrainOther500(LibriSpeech):

    url = "http://www.openslr.org/resources/12/train-other-500.tar.gz"
    dirname = 'LibriSpeech/train-other-500'
    filename = 'train-other-500.tar.gz'
