import torch
import unittest

from torchaudio import AudioDataLoader, SpectrogramDataset


class TestCases(unittest.TestCase):
    def test_loader(self):
        '''
        So firstly lets make sure that the dataloader returns the right shit
        '''

        audio_conf = dict(sample_rate=16000,
                          window_size=0.02,
                          window_stride=0.01,
                          window='hamming')
        labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        expected_audio_path = 'fake_audio.wav'
        expected_text_path = 'fake_audio.txt'
        ids = [
            [expected_audio_path, expected_text_path],
            [expected_audio_path, expected_text_path],
            [expected_audio_path, expected_text_path]
        ]

        input_samples = [
            (torch.randn(161, 500), 'FIRST'),
            (torch.randn(161, 1000), 'SECOND'),
            (torch.randn(161, 700), 'THIRD')
        ]
        train_dataset = SpectrogramDataset(audio_conf=audio_conf, labels=labels,
                                           normalize=False, ids=ids)

        def item_func(self, index):
            sample = self.ids[index]
            audio_path, transcript_path = sample[0], sample[1]
            return audio_path, transcript_path

        train_dataset.__getitem__ = item_func
        data = train_dataset[0]
        self.assertItemsEqual(data[0], expected_audio_path)
        self.assertItemsEqual(data[1], expected_text_path)

        train_dataset = SpectrogramDataset(audio_conf=audio_conf, labels=labels,
                                           normalize=False, ids=ids)

        def item_func(self, index):
            sample = input_samples[index]
            audio, transcript = sample[0], sample[1]
            return audio, transcript

        train_dataset.__getitem__ = item_func
        batch_size = len(input_samples)
        num_workers = 1
        train_loader = AudioDataLoader(train_dataset, batch_size=batch_size,
                                       num_workers=num_workers)
        data = iter(train_loader).next()


if __name__ == '__main__':
    unittest.main()
