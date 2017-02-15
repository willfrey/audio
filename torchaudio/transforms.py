"""Sub-package containing useful transforms for audio and text data."""
from __future__ import division

import types

import librosa
import numpy as np
import torch


# pylint: disable=R0903


##################
# Helper Classes #
##################

class _ArgHandler(object):
    # Class variable that specifies expected fields
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) != len(self._fields):
            raise TypeError(
                'Expected {} arguments'.format(len(self._fields)))

        # Set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # Set the additional arguments (if any)
        extra_args = kwargs.keys() - self._fields
        for name in extra_args:
            setattr(self, name, kwargs.pop(name))
        if kwargs:
            raise TypeError(
                'Duplicate values for {}'.format(','.join(kwargs)))


###############################
# General/Abstract Transforms #
###############################

class Transform(object):
    """Abstract base class for a transform.

    All other transforms should subclass it. All subclassses should
    override __call__ and __call__ should expect one argument, the
    data to transform, and return the transformed data.

    Any state variables should be stored internally in the __dict__
    class attribute and accessed through self, e.g., self.some_attr.
    """

    def __call__(self, data):
        raise NotImplementedError


class Compose(Transform):
    """Defines a composition of transforms.

    Data will be processed sequentially through each transform.

    Attributes:
        transforms: A list of Transforms to apply sequentially.
    """

    def __init__(self, transforms):
        """Inits Compose with the specified transforms.

        Args:
            transforms: Any iterable of transforms.
        """
        self._transforms = list(transforms)

    def __len__(self):
        return len(self._transforms)

    def __getitem__(self, index):
        return self._transforms[index]

    def __call__(self, data):
        for transform in self._transforms:
            data = transform(data)
        return data


class ToTensor(Transform):
    """Converts a numpy.ndarray to a torch.*Tensor."""

    def __call__(self, nparray):
        # pylint: disable=E1101
        return torch.from_numpy(nparray)
        # pylint: enable=E1101


class ToArray(Transform):
    """Converts a torch.*Tensor to a numpy.ndarray."""

    def __call__(self, tensor):
        return tensor.numpy()


class Lambda(Transform):
    """Applies a lamba as a transform.

    Attributes:
        func: A lambda function to be applied to data.
    """

    def __init__(self, func):
        """Inits Lambda with func."""
        assert isinstance(func, types.LambdaType)
        self.func = func

    def __call__(self, data):
        return self.func(data)


#############################
# Audio-oriented transforms #
#############################

class Resample(_ArgHandler, Transform):
    """Resample a time series from orig_sr to target_sr

       Attributes
       ----------
       orig_sr : number > 0 [scalar]
           original sampling rate of `y`

       target_sr : number > 0 [scalar]
           target sampling rate

       res_type : str
           resample type (see note)

           .. note::
               By default, this uses `resampy`'s high-quality mode
               ('kaiser_best'). If `res_type` is not recognized by
               `resampy.resample`, it then falls back on
               `scikits.samplerate` (if it is installed)

               If both of those fail, it will fall back on
               `scipy.signal.resample`.

               To force use of `scipy.signal.resample`, set
               `res_type='scipy'`.

       fix : bool
           adjust the length of the resampled signal to be of size
           exactly `ceil(target_sr * len(y) / orig_sr)`

       scale : bool
           Scale the resampled signal so that `y` and `y_hat` have
           approximately equal total energy.

       kwargs : additional keyword arguments
           If `fix==True`, additional keyword arguments to pass to
           `librosa.util.fix_length`.

       See Also
       --------
       librosa.util.fix_length
       scipy.signal.resample
       """
    _fields = ['orig_sr', 'target_sr']

    def __call__(self, y):
        """Resample a time series.

        Parameters
        ----------
        y : np.ndarray [shape=(n,) or shape=(2, n)]
        audio time series.  Can be mono or stereo.

        Returns
        -------
        y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
        """
        return librosa.resample(y, **self.__dict__)

# Feature


class STFT(_ArgHandler, Transform):
    """Short-time Fourier transform (STFT)

    Attributes
    ----------
    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number audio of frames between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    dtype       : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.
    """

    def __call__(self, y):
        """Short-time Fourier transform (STFT).

        Returns a real-valued matrix
        Returns a complex-valued matrix D such that
            `np.abs(D[f, t])` is the magnitude of frequency bin `f`
            at frame `t`

            `np.angle(D[f, t])` is the phase of frequency bin `f`
            at frame `t`

        Parameters
        ----------
        y : np.ndarray [shape=(n,)], real-valued
            the input signal (audio time series)

        Returns
        -------
        D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix

        """
        return librosa.stft(y, **self.__dict__)


class PowerSpectrogram(_ArgHandler, Compose):
    """Computes the power spectrogram of an input signal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transforms = [STFT(*args, **kwargs),
                            # pylint: disable=E1101
                            Lambda(lambda S: np.square(np.abs(S)))]

    # def __call__(self, y):
        # return super().__call__(y)


class LogAmplitude(_ArgHandler, Transform):
    """Returns the log-scaled amplitude of a spectrogram.

    This is used in LogPowerSpectrogram and LogMelSpectrogram.
    """

    def __call__(self, S):
        return librosa.logamplitude(S, **self.__dict__)


class LogPowerSpectrogram(_ArgHandler, Compose):
    """Computes the log-power spectrogram of an input signal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transforms = [
            PowerSpectrogram(*args, **kwargs),
            LogAmplitude(*args, **kwargs)
        ]

    # def __call__(self, y):
        # return super().__call__(y)


class MelSpectrogram(_ArgHandler, Transform):
    """Computes the Mel-scaled power spectrogram of an input signal."""

    def __call__(self, y):
        return librosa.feature.melspectrogram(y=y, **self.__dict__)


class LogMelSpectrogram(_ArgHandler, Compose):
    """Computes the log-power Mel spectrogram of an input signal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transforms = [
            MelSpectrogram(*args, **kwargs),
            LogAmplitude(*args, **kwargs)
        ]


class MFCC(_ArgHandler, Transform):
    """Computes the mel-frequency cepstral coefficients of an input signal."""

    def __call__(self, y):
        return librosa.feature.mfcc(y, **self.__dict__)


class StackMemory(_ArgHandler, Transform):
    """Short-term history embedding.

    Vertically concatenate a data vector or matrix with delayed
    copies of itself.
    """

    def __call__(self, data):
        return librosa.feature.stack_memory(data, **self.__dict__)


class TimeWarp(Transform):
    """Time-stretch an audio series by a fixed rate with some probability.

    The rate is selected randomly from between two values.

    It's just a jump to the left...
    """

    def __init__(self, rates=(1.0, 1.0), probability=0.0):
        self.rates = rates
        self.probability = probability

    def __call__(self, y):
        # pylint: disable=E1101
        success = np.random.binomial(1, self.probability)
        if success:
            rate = np.random.uniform(*self.rates)
            return librosa.effects.time_stretch(y, rate)
        return y


class PitchShift(Transform):
    """Pitch-shift the waveform by n_steps half-steps."""
    _fields = ['sr', 'n_steps']

    def __call__(self, y):
        return librosa.effects.pitch_shift(y, **self.__dict__)


# pylint: disable=C0103, R0913
class InjectNoise(Transform):
    """Adds noise to an input signal with some probability and some SNR.

    Only adds a single noise file from the list right now.
    """

    def __init__(self,
                 path=None,
                 paths=None,
                 sr=16000,
                 noise_levels=(0, 0.5),
                 probability=0.4):
        if path:
            self.paths = librosa.util.find_files(path)
        else:
            self.paths = paths
        self.sr = sr
        self.noise_levels = noise_levels
        self.probability = probability

    def __call__(self, data):
        # pylint: disable=E1101
        success = np.random.binomial(1, self.probability)
        if success:
            noise_src, _ = librosa.load(
                np.random.choice(self.paths), sr=self.sr)
            noise_offset_fraction = np.random.rand()
            noise_level = np.random.uniform(*self.noise_levels)

            noise_dst = np.zeros_like(data)

            src_offset = int(len(noise_src) * noise_offset_fraction)
            src_left = len(noise_src) - src_offset

            dst_offset = 0
            dst_left = len(data)

            while dst_left > 0:
                copy_size = min(dst_left, src_left)
                np.copyto(noise_dst[dst_offset:dst_offset + copy_size],
                          noise_src[src_offset:src_offset + copy_size])
                if src_left > dst_left:
                    dst_left = 0
                else:
                    dst_left -= copy_size
                    dst_offset += copy_size
                    src_left = len(noise_src)
                    src_offset = 0

            data += noise_level * noise_dst

        return data

    def __getitem__(self, index):
        return self.paths[index]


class SwapSamples(Transform):
    """Intentionally corrupts mono audio by swapping adjacent samples.

    Iterates over each sample (excluding the boundary) of the input
    signal and will stochastically swap the sample with its neighbor.
    Whether the neighbor is the left or right sample is chosen randomly
    with equal probability.
    """

    def __init__(self, probability):
        self.probability = probability

    def __call__(self, y):
        # Vectorize this?
        for i in range(1, len(y) - 1):
            # pylint: disable=E1101
            success = np.random.binomial(1, self.probability)
            if success:
                swap_right = np.random.binomial(1, 0.5)
                if swap_right:
                    y[i], y[i + 1] = y[i + 1], y[i]
                else:
                    y[i], y[i - 1] = y[i - 1], y[i]
        return y


class TimeShift(Transform):
    """Stochastically shifts audio samples."""

    def __init__(self, probability, min_max_displacement=(80, 81)):
        self.probability = probability
        self.min_max_displacement = min_max_displacement

    def __call__(self, y):
        # pylint: disable=E1101
        success = np.random.binomial(1, self.probability)
        if success:
            shift = np.random.randint(*self.min_max_displacement)
            if np.random.binomial(1, 0.5):  # shift to the right
                y[shift:] = y[:-shift]
                y[:shift] = 0
            else:
                y[:-shift] = y[shift:]
                y[-shift:] = 0

        return y


############################
# Text-Oriented Transforms #
############################

class CharToInt(Transform):
    """Maps a string or other iterable, character-wise, to integer labels.

    Attributes:
        char_to_index: A dictionary containing character keys and
        integer values.

    """

    def __init__(self, char_to_index):
        self.char_to_index = char_to_index

    def __call__(self, data):
        return [int_ for int_ in map(self.char_to_index, data)
                if int_ is not None]
