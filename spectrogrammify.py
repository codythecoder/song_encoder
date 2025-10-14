import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# filepath = r'C:\Users\cody.lovett\Music\Patricia Taxxon - TECHDOG 1-7\Patricia Taxxon - TECHDOG 1-7 - 36 GDGEGDGCDEDHECETCHCOHTHOTOTO.mp3'
# filepath = r'C:\Users\cody.lovett\Music\dgh06 (Some other sort of House) [Projectfile](1).wav'
filepath = r'C:\Users\cody.lovett\Music\Julian Gray feat. SOFI - Revolver (Edge Split Remix) v2.wav'
# filepath = r'C:\Users\cody.lovett\Music\13. rain mist, snow mist, fire that follows-v04.wav'

sample_rate, samples = wavfile.read(filepath)
if (len(samples.shape) == 2):
    samples = samples[:, 0]

# get the first few seconds
# scale it down to something readable
samples = samples[sample_rate*10:sample_rate*35]
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, scaling='spectrum')

import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

# Sxx = spectrogram
Sxx = 10*numpy.log10(spectrogram)

plt.pcolormesh(times, frequencies, Sxx)
# plt.pcolormesh(times, frequencies, spectrogram)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_yscale('symlog')
plt.axis('scaled')
# plt.imshow(spectrogram, extent=(min(times),max(times),max(frequencies),min(frequencies)), aspect='auto')
plt.imshow(Sxx, extent=(min(times),max(times),max(frequencies),min(frequencies)), aspect='auto', )
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.waitforbuttonpress()