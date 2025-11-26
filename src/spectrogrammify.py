from typing import Any, Optional

import warnings

from scipy import signal
from scipy.io import wavfile

import pydub

import numpy as np
import numpy.typing as npt

from src.encoder_types import Config



def spectrogrammify(
        filepath: str,
        config: Config,
        audio_channel: int = 0,
        nfft: Optional[int] = None,
        noverlap: int = 0,
        ) -> npt.NDArray[np.float64]:

    beginning_silence = config['SPECTROGRAM_NPERSEG'] - noverlap

    sample_rate, samples = load_audiofile(filepath)
    sample_dtype = samples.dtype

    # only use one channel of audio
    if (len(samples.shape) == 2):
        samples = samples[:, audio_channel]

    # don't resample audio up
    if sample_rate < config['EXPECTED_BITRATE']:
        raise ValueError(f'Sample rate is too low. Expected {config["EXPECTED_BITRATE"]}, got {sample_rate}.')

    # resample down if necessary
    if sample_rate > config['EXPECTED_BITRATE']:
        samples = resample_audio(samples, sample_rate, config['EXPECTED_BITRATE'])
        sample_rate = config['EXPECTED_BITRATE']


    # add a blip to the beginning of the audio
    #   this forces the frequency buckets to always be in the same range
    if sample_dtype == np.int16:
        samples = np.concatenate([
            [2**15-1,-2**15]+[0]*(beginning_silence-2),
            samples
        ])
    elif sample_dtype == np.int32:
        samples = np.concatenate([
            [2**31-1,-2**31]+[0]*(beginning_silence-2),
            samples
        ])
    else:
        raise NotImplementedError('Audio bit depth not process correctly. Got ' + str(sample_dtype))

    frequencies, times, spectrogram = signal.spectrogram(
        samples,
        sample_rate,
        nperseg=config['SPECTROGRAM_NPERSEG'],
        nfft=nfft,
        noverlap=noverlap,
        scaling='spectrum',
    )

    # print(f'{nperseg=}')
    # print(f'{nfft=}')
    # print(f'{noverlap=}')
    # print(f'{len(frequencies)=}')
    # print(f'{len(samples)=}')
    # print(f'{len(times)=}')
    # print(f'{sample_rate=}')
    # print(f'samples_per_second={len(times)/(len(samples)/sample_rate):.2f}')

    return spectrogram[:,1:]  # type: ignore


def load_audiofile(filepath: str) -> tuple[int, npt.NDArray[Any]]:
    if filepath.endswith('.wav'):
        return load_audiofile_wav(filepath)
    elif filepath.endswith('.mp3'):
        return load_audiofile_mp3(filepath)
    else:
        raise NotImplementedError


def load_audiofile_wav(filepath: str) -> tuple[int, npt.NDArray[Any]]:
    with warnings.catch_warnings(action="ignore"):
        sr, s = wavfile.read(filepath)
    return sr, s


def load_audiofile_mp3(filepath: str) -> tuple[int, npt.NDArray[Any]]:
    audio = pydub.AudioSegment.from_mp3(filepath)
    y = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        y = y.reshape((-1, 2))

    return audio.frame_rate, y


def resample_audio(
        samples: npt.NDArray[Any],
        curr_sample_rate: int,
        new_sample_rate: int,
        ) -> npt.NDArray[np.float32]:

    output = signal.resample_poly(
        samples.astype(np.float32),
        new_sample_rate,
        curr_sample_rate,
    )

    return output  # type: ignore