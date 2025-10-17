from typing import Optional
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.io import wavfile

# filepath = r'C:\Users\cody.lovett\Music\Patricia Taxxon - TECHDOG 1-7\Patricia Taxxon - TECHDOG 1-7 - 36 GDGEGDGCDEDHECETCHCOHTHOTOTO.mp3'
# filepath = r'C:\Users\cody.lovett\Music\dgh06 (Some other sort of House) [Projectfile](1).wav'
# filepath = r'C:\Users\cody.lovett\Music\Julian Gray feat. SOFI - Revolver (Edge Split Remix) v2.wav'
filepath = r'C:\Users\cody.lovett\Music\13. rain mist, snow mist, fire that follows-v04.wav'

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

    print(
        nperseg,
        nfft,
        noverlap,
        len(frequencies),
        f'{len(times)/(len(samples)/sample_rate):.2f}'
    )

    return frequencies, times[1:], spectrogram[:,1:]



if __name__ == '__main__':
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