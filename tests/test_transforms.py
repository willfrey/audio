"""Unit tests for torchaudio.transforms."""

from unittest import TestCase

import numpy as np
import torch

from torchaudio import transforms

class TransformTest(TestCase):

    def set_up(self):
        self.transform = transforms.Transform()

    def test_callable(self):
        self.assertTrue(callable(self.transform))

class ToTensorTest(TestCase):

    def set_up(self):
        self.to_tensor = transforms.ToTensor()

    def test_to_tensor(self):
        nparray = np.array([1, 2, 3])
        tensor = self.to_tensor(nparray)
        self.assertIsInstance(tensor, torch.Tensor)
