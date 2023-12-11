import librosa
import os


import numpy as np
import matplotlib.pyplot as plt

audio_path = os.fspath("WP_A_Hel.wav")
x, fs = librosa.load(audio_path)

deltaf = 5
frame_size = fs // deltaf
hop_size = frame_size // 2
f0, voiced_flag, voiced_probs = librosa.pyin(x,fmin=50,fmax=800,sr=fs,frame_length=frame_size,hop_length=hop_size)
print(f0)
print(voiced_flag)
times = librosa.times_like(f0)

D = librosa.amplitude_to_db(np.abs(librosa.stft(x,hop_length=hop_size,n_fft=frame_size)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')


y, sr = librosa.load(librosa.ex('trumpet'))
f0, voicing, voicing_p = librosa.pyin(y=y, sr=sr, fmin=200, fmax=700)
S = np.abs(librosa.stft(y))
freqs = librosa.fft_frequencies(sr=sr)
harmonics = np.arange(1, 13)
f0_harm = librosa.f0_harmonics(S, freqs=freqs, f0=f0, harmonics=harmonics)
bsdagjkjdfsknlgfmkl
kjbsgrfndklkl