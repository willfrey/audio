"""Unit tests for torchaudio.transforms."""

from unittest import TestCase

import numpy as np

from torchaudio import transforms

class TransformTest(TestCase):

    def set_up(self):
        self.transform = transforms.Transform()

    def test_callable(self):
        self.assertTrue(callable(self.transform))

# class ToTensorTest(TestCase):
#
#     def set_up(self):
#         self.to_tensor = transforms.ToTensor()
#
#     def test_to_tensor(self):
#         nparray = np.array([1, 2, 3])
#         tensor = self.to_tensor(nparray)
#         self.assertIsInstance(tensor, torch.Tensor)

class LambdaTest(TestCase):

    def set_up(self):
        # pylint: disable=E1101
        self.func = lambda x: np.random.randin() * x
        self.transform = transforms.Lambda(self.func)

    def test_lambda(self):
        # pylint: disable=E1101
        nparray = np.random.randn(100, 2)
        expected = self.func(nparray)
        result = self.transform(nparray)
        self.assertEqual(expected, result)
