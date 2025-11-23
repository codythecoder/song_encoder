from typing import Optional
import warnings
from scipy import signal
import numpy as np
import numpy.typing as npt
from scipy.io import wavfile


def spectrogrammify(
        filepath: str,
        audio_channel: int = 0,
        nperseg: int = 256,
        nfft: Optional[int] = None,
        noverlap: int = 0,
        ) -> tuple[
            int,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]]:

    beginning_silence = nperseg - noverlap

    # print(filepath)
    with warnings.catch_warnings(action="ignore"):
        sample_rate, samples = wavfile.read(filepath)
    # print(filepath)
    if (len(samples.shape) == 2):
        # just one ear
        samples = samples[:, audio_channel]

    # force the frequencies to be in a consistent range by adding a blip to the start
    samples = np.concatenate([
        [2**15-1,-2**15]+[0]*(beginning_silence-2),
        samples
    ])

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, nfft=nfft, noverlap=noverlap, scaling='spectrum')

    # print(f'{nperseg=}')
    # print(f'{nfft=}')
    # print(f'{noverlap=}')
    # print(f'{len(frequencies)=}')
    # print(f'{len(samples)=}')
    # print(f'{len(times)=}')
    # print(f'{sample_rate=}')
    # print(f'samples_per_second={len(times)/(len(samples)/sample_rate):.2f}')

    return sample_rate, frequencies, times[1:], spectrogram[:,1:]

