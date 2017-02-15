"""Generic audio datasets and helper functions."""
from __future__ import print_function

import os
import os.path

import torch.utils.data

import torchaudio.utils


# pylint: disable=R0903
class TranscriptionDataset(torch.utils.data.Dataset):
    """Dataset for transcription tasks."""

    def __init__(self, utts,
                 transform=None,
                 target_transform=None,
                 loader=torchaudio.utils.load_audio):

        self.utts = utts
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, key):
        path, target = self.utts[key]
        utt, _ = self.loader(path)
        if self.transform is not None:
            utt = self.transform(utt)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return utt, target

    def __len__(self):
        return len(self.utts)


# pylint: disable=R0903
class TarDataset(object):
    """An abstract class representing a dataset from a tarball.

    Attributes:
        url: URL where the tar or tar.gz archive can be downloaded.
        filename: Filename of the downloaded tarball.
        dirname: Name of the top-level directory within the tarball
            that contains the data files.
    """
    dirname = None
    filename = None
    url = None

    @classmethod
    def download_or_untar(cls, root):
        """Downloads and untars the tarball located at `url` to the
        directory specified by `root`"""
        import tarfile
        from six.moves import urllib

        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('Downloading {cls.url}'.format(cls=cls))
                with open(tpath, 'wb') as tfile:
                    tdata = urllib.request.urlopen(cls.url)
                    tfile.write(tdata.read())
            print('Extracting {cls.filename} to {cls.dirname}'.format(cls=cls))
            with tarfile.open(tpath, 'r') as tfile:
                tfile.extractall(path=root)
            print('Removing {cls.filename}'.format(cls=cls))
            os.unlink(tpath)
        return os.path.join(path, '')
