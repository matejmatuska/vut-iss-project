import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk, buttord, butter

def dft(x):
    n = np.arange(1024)
    k = n.reshape((1024, 1))
    bases = np.exp(-2j * np.pi * n * k / 1024)
    # multiply the bases matrix with the signal vector
    return np.dot(bases, x)

def plot_dft(dft, fs, N, title):
    x_axis = np.arange(N) / N * fs / 2
    y_axis = np.abs(dft[:N])

    plt.figure(figsize=(6,5))
    plt.gca().set_xlabel('$f[Hz]$')
    plt.gca().set_title(title)
    plt.plot(x_axis, y_axis)
    plt.show(block = True)

def find_noise_freqs(dft, N, fs):
    indices = [i for i,v in enumerate(np.abs(dft)) if v > 10]
    freqs = [(i / N * fs) for i in indices]
    return freqs

def gen_specgram(x, fs):
    f, t, sgr = spectrogram(x, fs, nperseg=1024, noverlap=512)
    sgr_log = 10 * np.log10(sgr+1e-20 ** 2)

    plt.figure(figsize=(9,4))
    plt.pcolormesh(t, f, sgr_log)
    plt.gca().set_xlabel('$t[s]$')
    plt.gca().set_ylabel('$f[Hz]$')
    plt.show(block = True)

def gen_cos(f, fs, size):
    t = np.arange(size) / fs
    x = np.cos((f * 2 * np.pi) * t)
    return x

sig, SAMPLE_RATE = sf.read('xmatus36.wav')

# zaklady
print("Length:")
print(sig.size / SAMPLE_RATE)
print("Length in samples:")
print(sig.size)
print("Samplerate:")
print(SAMPLE_RATE)

maxval = np.max(sig)
minval = np.min(sig)
print("Max hodnota:")
print(maxval)
print("Min hodnota:")
print(minval)

t = np.arange(sig.size) / SAMPLE_RATE
plt.figure(figsize=(6,3))
plt.plot(t, sig)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Vstupný signál')
plt.tight_layout()
plt.show(block=True)

# ustrednenie
mean = np.mean(sig)
sig = sig - mean

# normalize
maxi = max(maxval, minval, key=abs)
sig = sig / maxi

plt.figure(figsize=(6,3))
plt.plot(t, sig)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Centralizovaný a normalizovaný zvukový signál')
plt.tight_layout()
plt.show(block=True)

N = 1024
step = 512
frames = [sig[i : i + N] for i in range(0, len(sig), step)]
frame = frames[28]

# plot chosen frame
#for i in range(0, len(frames)):
t = np.arange(frame.size) / SAMPLE_RATE
plt.figure(figsize=(10,5))
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('29. rámec vstupného signálu')
plt.plot(t, frame)
plt.show()

frame_dft = dft(frame)
plot_dft(frame_dft, SAMPLE_RATE, 512, 'Modul DFT 29. rámca')

gen_specgram(sig, SAMPLE_RATE)

detection_dft = dft(frames[1]);
plot_dft(detection_dft, SAMPLE_RATE, 512, "Modul DFT rámca č. 2")

noise_freqs = find_noise_freqs(detection_dft[:512], 512, SAMPLE_RATE / 2)
cos = gen_cos(noise_freqs[0], SAMPLE_RATE, sig.size)
cos += gen_cos(noise_freqs[1], SAMPLE_RATE, sig.size)
cos += gen_cos(noise_freqs[2], SAMPLE_RATE, sig.size)
cos += gen_cos(noise_freqs[3], SAMPLE_RATE, sig.size)
gen_specgram(cos, SAMPLE_RATE)

sf.write('audio/4cos.wav', cos, SAMPLE_RATE)

def butter_bandstop_filter(sig, f, fs):
    nyq_fs = 0.5 * fs

    wp = [(f - 65) / nyq_fs, (f + 65) / nyq_fs]
    ws = [(f - 15) / nyq_fs, (f + 15) / nyq_fs]

    ord, wn = buttord(wp, ws, 3, 40, False);
    b, a = butter(ord, wn, btype='bandstop')
    y = lfilter(b, a, sig)
    return y

for i in range(len(noise_freqs)):
    # filter out each noise cosine frequency
    sig = butter_bandstop_filter(sig, noise_freqs[i], SAMPLE_RATE);
gen_specgram(sig, SAMPLE_RATE)

# write filtered signal to file
sf.write('audio/clean_bandstop.wav', sig, SAMPLE_RATE)

