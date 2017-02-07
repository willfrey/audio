"""AN4 Dataset"""
from __future__ import print_function

import os
import os.path
import re

from torchaudio import data


def _make_dataset(path, split):
    # pylint: disable=W1401
    pattern = re.compile('^(<[^>]+>)?\s?(?P<text>.*?)\s(<[^>]+>)?\s?\((.*)\)$')
    dirname = os.path.join(path, 'etc')
    basename = 'an4_' + split
    with open(os.path.join(dirname, basename + '.transcription')) as text_f:
        texts = (pattern.search(line).group('text')
                 for line in text_f.readlines())
    with open(os.path.join(dirname, basename + '.fileids')) as utt_f:
        paths = (os.path.join(path, 'wav', line.strip() + '.sph')
                 for line in utt_f.readlines())
    utterances = list(zip(paths, texts))
    return utterances


# pylint: disable=R0903
class AN4(data.TarDataset, data.TranscriptionDataset):
    """AN4 Dataset"""

    url = 'http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz'
    dirname = 'an4'
    filename = 'an4_sphere.tar.gz'

    @classmethod
    def train(cls, root, **kwargs):
        """Loads the train split of AN4."""
        path = cls.download_or_untar(root)
        utts = _make_dataset(path, 'train')
        return cls(utts=utts, **kwargs)

    @classmethod
    def test(cls, root, **kwargs):
        """Loads the test split of AN4."""
        path = cls.download_or_untar(root)
        utts = _make_dataset(path, 'test')
        return cls(utts=utts, **kwargs)
