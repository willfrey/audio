from __future__ import print_function
import torch.utils.data as data
import librosa
import os
import os.path
import errno
import torch
import numpy as np


class AN4(data.Dataset):
    url = 'http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz'
    base_folder = 'an4'
    processed_folder = 'processed'

    def __init__(self,
                 root,
                 sample_rate=16000,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = root
        self.sample_rate = sample_rate

        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        def an4_sph_filename(s):
            return os.path.join(self.root, self.base_folder, 'wav',
                                '{}.sph'.format(s.strip()))

        # TODO: clean these up into concise functions/methods
        if self.train:
            with open(
                    os.path.join(self.root, self.base_folder, 'etc',
                                 'an4_train.fileids')) as f:
                self.train_data = [
                    torch.from_numpy(
                        librosa.load(
                            an4_sph_filename(line), sr=self.sample_rate)[0])
                    for line in f
                ]

            with open(
                    os.path.join(self.root, self.base_folder, 'etc',
                                 'an4_train.transcription')) as f:
                self.train_labels = [
                    line.rpartition(' ')[0].replace('<s>', '').replace(
                        '</s>', '').strip() for line in f
                ]
        else:
            with open(
                    os.path.join(self.root, self.base_folder, 'etc',
                                 'an4_test.fileids')) as f:
                self.train_data = [
                    torch.from_numpy(
                        librosa.load(
                            an4_sph_filename(line), sr=self.sample_rate)[0])
                    for line in f
                ]

            with open(
                    os.path.join(self.root, self.base_folder, 'etc',
                                 'an4_test.transcription')) as f:
                self.test_labels = [
                    line.rpartition(' ')[0].strip() for line in f
                ]

    def __getitem__(self, index):
        if self.train:
            audio, transcript = self.train_data[index], self.train_labels[
                index]
        else:
            audio, transcript = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            transcript = self.target_transform(transcript)

        return audio, transcript

    def __len__(self):
        if self.train:
            return 948
        else:
            return 130

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def download(self):
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        data = urllib.request.urlopen(self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        with tarfile.open(file_path, 'r:gz') as tar_f:
            tar_f.extractall(path=os.path.join(self.root))
        os.unlink(file_path)

    # TODO: Maybe serialize these into pickle files for faster loading?
