from typing import Optional
from scipy import signal
import numpy as np
from scipy.io import wavfile


def spectrogrammify(
        filepath: str,
        audio_channel: int = 0,
        nperseg=256,
        nfft: Optional[int] = None,
        noverlap=0):

    beginning_silence = nperseg - noverlap

    sample_rate, samples = wavfile.read(filepath)
    if (len(samples.shape) == 2):
        # just one ear
        samples = samples[:, audio_channel]

    # force the frequencies to be in a consistent range by adding a blip to the start
    samples = np.concatenate([
        [2**15-1,-2**15]+[0]*(beginning_silence-2),
        samples
    ])

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, nfft=nfft, noverlap=noverlap, scaling='spectrum')

    print(f'{nperseg=}')
    print(f'{nfft=}')
    print(f'{noverlap=}')
    print(f'{len(frequencies)=}')
    print(f'{len(samples)=}')
    print(f'{len(times)=}')
    print(f'{sample_rate=}')
    print(f'samples_per_second={len(times)/(len(samples)/sample_rate):.2f}')

    return frequencies, times[1:], spectrogram[:,1:]



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    filepath = r'13. rain mist, snow mist, fire that follows-v04.wav'
    frequencies, times, spectrogram = spectrogrammify(filepath, nperseg=1024)
    times = times[:1000]
    spectrogram = spectrogram[:,:1000]

    import sys
    np.set_printoptions(threshold=sys.maxsize)

    # Sxx = spectrogram
    Sxx = 10*np.log10(spectrogram)

    plt.pcolormesh(times, frequencies, Sxx)
    # plt.pcolormesh(times, frequencies, spectrogram)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # ax.set_yscale('symlog')
    # plt.axis('scaled')
    # plt.imshow(spectrogram, extent=(min(times),max(times),max(frequencies),min(frequencies)), aspect='auto')
    plt.imshow(Sxx, extent=(min(times),max(times),max(frequencies),min(frequencies)), aspect='auto', )
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.waitforbuttonpress()