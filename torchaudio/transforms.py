"""A number of useful transforms for audio data."""
from __future__ import division

import types
from functools import wraps

import librosa
import numpy as np
import torch


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class Lambda(object):
    """Applies a lamba as a tranform."""

    def __init__(self, lambda_):
        assert isinstance(lambda_, types.LambdaType)
        self.lambda_ = lambda_

    def __call__(self, inp):
        return self.lambda_(inp)


class ToTensor(object):
    """Converts a numpy.ndarray to a torch.*Tensor."""

    def __call__(self, array):
        # pylint: disable=E1101
        return torch.from_numpy(array)


class ToArray(object):
    """Converts a torch.*Tensor to a numpy.ndarray."""

    def __call__(self, tensor):
        return tensor.numpy()


# class _FunctionFaker(object):
#
#     def __init__(self, func):
#         wraps(func)(self)
#
#     def __call__(self, *args, **kwargs):
#         # pylint: disable=E1101
#         return self.__wrapped__(*args, **kwargs)
#
#     def __get__(self, instance, cls):
#         if instance is None:
#             return self
#         else:
#             return types.MethodType(self, instance)


class _FuncArgsHandler(object):
    # Class variable that specifies expected fields
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # Set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # Set the additional arguments (if any)
        extra_args = kwargs.keys() - self._fields
        for name in extra_args:
            setattr(self, name, kwargs.pop(name))
        if kwargs:
            raise TypeError('Duplicate values for {}'.format(','.join(kwargs)))


class Resample(_FuncArgsHandler):
    _fields = ['orig_sr', 'targ_sr']

    def __doc__(self):
        return librosa.resample.__doc__

    def __call__(self, signal):
        return librosa.resample(signal, **self.__dict__)


class STFT(_FuncArgsHandler):

    def __call__(self, signal):
        return librosa.stft(signal, **self.__dict__)


class PowerSpectrogram(_FuncArgsHandler):

    def __call__(self, stft_matrix):
        return np.abs(stft_matrix)**2


class LogAmplitude(_FuncArgsHandler):

    def __call__(self, spect):
        return librosa.logamplitude(spect, **self.__dict__)


class MelSpectrogram(_FuncArgsHandler):
    # pylint: disable=C0103

    def __call__(self, y=None, S=None):
        return librosa.feature.melspectrogram(y=y, S=S, **self.__dict__)


class MFCC(_FuncArgsHandler):
    # pylint: disable=C0103

    def __call__(self, y=None, S=None):
        return librosa.feature.mfcc(y=y, S=S, **self.__dict__)


class Frame(_FuncArgsHandler):

    def __call__(self, signal):
        return librosa.util.frame(signal, **self.__dict__)


class TextToInt(object):

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, text):
        return [k for k in map(self.mapping.get, text) if k is not None]
