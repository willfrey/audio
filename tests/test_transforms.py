"""Unit tests for torchaudio.transforms."""

from unittest import TestCase

from torchaudio import transforms

class TransformTest(TestCase):

    def set_up(self):
        self.transform = transforms.Transform()

    def test_callable(self):
        self.assertTrue(callable(self))
