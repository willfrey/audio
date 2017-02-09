"""Unit tests for torchaudio.transforms."""

import random
import unittest

import torch

import torchaudio.transforms


class PipelineSingletonTestCase(unittest.TestCase):
    """Tests a singleton transform."""

    def setUp(self):
        self.transforms = torchaudio.transforms.Transform()
        self.args = [torchaudio.transforms.Transform(),
                     torchaudio.transforms.Transform()]

    def test_init(self):
        """Tests init with singleton argument."""
        pipeline = torchaudio.transforms.Pipeline(self.transforms)
        print()
        for transform in pipeline:
            print(transform)

        pipeline = torchaudio.transforms.Pipeline(self.transforms, self.args)
        print()
        for transform in pipeline:
            print(transform)

        pipeline = torchaudio.transforms.Pipeline(self.transforms, *self.args)
        print()
        for transform in pipeline:
            print(transform)

        pipeline = torchaudio.transforms.Pipeline(
            self.transforms, self.args[0])
        print()
        for transform in pipeline:
            print(transform)


class PipelineListTestCase(unittest.TestCase):
    """Tests a singleton transform."""

    def setUp(self):
        self.transforms = [torchaudio.transforms.Transform(),
                           torchaudio.transforms.Transform()]
        self.args = [torchaudio.transforms.Transform(),
                     torchaudio.transforms.Transform()]

    def test_init(self):
        """Tests init with list argument."""
        pipeline = torchaudio.transforms.Pipeline(self.transforms)
        print()
        for transform in pipeline:
            print(transform)

        pipeline = torchaudio.transforms.Pipeline(self.transforms, self.args)
        print()
        for transform in pipeline:
            print(transform)

        pipeline = torchaudio.transforms.Pipeline(self.transforms, *self.args)
        print()
        for transform in pipeline:
            print(transform)

        pipeline = torchaudio.transforms.Pipeline(
            self.transforms, self.args[0])
        print()
        for transform in pipeline:
            print(transform)


if __name__ == '__main__':
    unittest.main()
