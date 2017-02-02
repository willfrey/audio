import os
import io
import shutil

import subprocess

from torchaudio import SpectrogramDataset
from torchaudio.utils import create_manifest


class AN4(SpectrogramDataset):
    def __init__(self, audio_conf, labels, normalize=False, classes='train', dataset_path='./an4_dataset'):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param classes: Path to manifest csv as describe above
        :param dataset_path: Path to manifest csv as describe above
        """
        train_manifest_path = 'an4_train_manifest.csv'
        test_manifest_path = 'an4_test_manifest.csv'
        if not (os.path.isfile(train_manifest_path) and os.path.isfile(test_manifest_path)):
            print ('Could not find cached manifests, creating manifests...')
            sample_rate = audio_conf['sample_rate']
            name = 'an4'
            root_path = 'an4/'
            subprocess.call(['wget http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz'], shell=True)
            subprocess.call(['tar -xzvf an4_raw.bigendian.tar.gz'], stdout=open(os.devnull, 'wb'), shell=True)
            os.makedirs(dataset_path)
            _format_data(root_path, dataset_path, 'train', name, 'an4_clstk', sample_rate)
            _format_data(root_path, dataset_path, 'test', name, 'an4test_clstk', sample_rate)
            shutil.rmtree(root_path)
            os.remove('an4_raw.bigendian.tar.gz')
            train_path = dataset_path + '/train/'
            test_path = dataset_path + '/test/'
            if not os.path.isfile(train_manifest_path):
                create_manifest(train_path, train_manifest_path)
            if not os.path.isfile(test_manifest_path):
                create_manifest(test_path, test_manifest_path)
        if classes == 'train':
            manifest_filepath = train_manifest_path
        elif classes == 'test':
            manifest_filepath = test_manifest_path
        else:
            RuntimeError('Classes either has to be test or train for AN4')
        super(AN4, self).__init__(audio_conf, manifest_filepath, labels, normalize=normalize)


def _format_data(root_path, new_data_path, data_tag, name, wav_folder, sample_rate):
    data_path = new_data_path + data_tag + '/' + name + '/'
    new_transcript_path = data_path + '/txt/'
    new_wav_path = data_path + '/wav/'

    os.makedirs(new_transcript_path)
    os.makedirs(new_wav_path)

    wav_path = root_path + 'wav/'
    file_ids = root_path + 'etc/an4_%s.fileids' % data_tag
    transcripts = root_path + 'etc/an4_%s.transcription' % data_tag
    train_path = wav_path + wav_folder

    _convert_audio_to_wav(train_path, sample_rate)
    _format_files(file_ids, new_transcript_path, new_wav_path, transcripts, wav_path)


def _convert_audio_to_wav(train_path, sample_rate):
    with os.popen('find %s -type f -name "*.raw"' % train_path) as pipe:
        for line in pipe:
            raw_path = line.strip()
            new_path = line.replace('.raw', '.wav').strip()
            cmd = 'sox -t raw -r %d -b 16 -e signed-integer -B -c 1 \"%s\" \"%s\"' % (
                sample_rate, raw_path, new_path)
            os.system(cmd)


def _format_files(file_ids, new_transcript_path, new_wav_path, transcripts, wav_path):
    with open(file_ids, 'r') as f:
        with open(transcripts, 'r') as t:
            paths = f.readlines()
            transcripts = t.readlines()
            for x in range(len(paths)):
                path = wav_path + paths[x].strip() + '.wav'
                filename = path.split('/')[-1]
                extracted_transcript = _process_transcript(transcripts, x)
                current_path = os.path.abspath(path)
                new_path = new_wav_path + filename
                text_path = new_transcript_path + filename.replace('.wav', '.txt')
                with io.FileIO(text_path, "w") as file:
                    file.write(extracted_transcript)
                os.rename(current_path, new_path)


def _process_transcript(transcripts, x):
    extracted_transcript = transcripts[x].split('(')[0].strip("<s>").split('<')[0].strip().upper()
    return extracted_transcript
