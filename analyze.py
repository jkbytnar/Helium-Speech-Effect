import glob
import os
from scipy.io.wavfile import read, WavFileWarning
import pandas as pd
import librosa
import re
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import warnings

def comparison(audio_path: str, get_spec: bool, get_mfcc: bool):
    data = get_audio(audio_path)

    if get_spec: plot_spec(data)
    if get_mfcc: plot_mfcc(data)

    return data

def get_audio(audio_path: str):
    audio_path = audio_path + "\*.wav"
    data = pd.DataFrame(columns=['Speaker', 'Content', 'Hel', 'Signal'])
    pattern = re.compile(r'(\w+)_(\w+)_(\w+).wav')

    for path in glob.glob(pathname=audio_path):
        basename = os.path.basename(path)

        warnings.filterwarnings("ignore", category=WavFileWarning)
        fs, x = read(path)

        if fs != 16000:
            x = librosa.resample(x.astype(float), orig_sr=fs, target_sr=16000)

        match = pattern.match(basename)
        if match:
            speaker = match.group(1)
            content = match.group(2)
            hel = True if match.group(3) == "Hel" else False
            row = {'Speaker': speaker, 'Content': content, 'Hel': hel, 'Signal': x}
            data.loc[len(data)] = row

    return data


def plot_spec(data):
    fs = 16000
    deltaf = 5
    frame_size = fs // deltaf
    max_f = 1000
    args = dict(window='hamming', nperseg=frame_size, noverlap=0,
                                        scaling='spectrum',
                                        mode='complex', detrend='constant')
    path = "./Plots/Spectrograms"
    if not os.path.exists(path):
        os.mkdir(path)

    for index, row in data[data['Hel'] == False].iterrows():
        figure, ax = plt.subplots(2, 1, figsize=(10, 14))

        row_hel = data[(data['Hel'] == True) &
                       (data['Content'] == row['Content']) &
                       (data['Speaker'] == row['Speaker'])]

        if len(row['Signal']) > len(row_hel['Signal'].iloc[0]):
            s_nohel = (row['Signal'])[:(len(row_hel['Signal'].iloc[0]))]
            s_hel = row_hel['Signal'].iloc[0]
        else:
            s_nohel = row['Signal']
            s_hel = (row_hel['Signal'].iloc[0])[:len(row['Signal'])]

        f_nohel, t_nohel, Spec_nohel = signal.spectrogram(s_nohel, fs, **args)
        f_hel, t_hel, Spec_hel = signal.spectrogram(s_hel, fs, **args)

        Spec_nohel = Spec_nohel / np.max(abs(Spec_nohel))
        Spec_hel = Spec_hel / np.max(abs(Spec_hel))

        im = ax[0].pcolormesh(t_nohel, f_nohel, abs(Spec_nohel))
        ax[0].set_ylim(0, max_f)
        ax[0].set_yticks(np.arange(0, max_f + 1, max_f / 20))
        ax[0].set_title(f"{row['Speaker']}, {row['Content']}, 'NoHel")
        ax[0].set_xlabel("Time [sec]")
        ax[0].set_ylabel("Frequency [Hz]")
        figure.colorbar(im, ax=ax[0])

        im = ax[1].pcolormesh(t_hel, f_hel, abs(Spec_hel))
        ax[1].set_ylim(0, max_f)
        ax[1].set_yticks(np.arange(0, max_f + 1, max_f / 20))
        ax[1].set_title(f"{row['Speaker']}, {row['Content']}, 'Hel")
        ax[1].set_xlabel("Time [sec]")
        ax[1].set_ylabel("Frequency [Hz]")
        figure.colorbar(im, ax=ax[1])

        plt.savefig("./Plots/Spectrograms/" + f"Spectrogram_{row['Speaker']}_{row['Content']}.png")
        plt.close()
    return


def plot_mfcc(data):
    args = {
    'n_fft': 512,             # Liczba prążków w STFT
    'win_length': 320,        # Długość okna
    'hop_length': 160,        # Interwał pomiędzy oknami (przeskok)
    'n_mels': 30,             # Liczba filtrów melowych
    'n_mfcc': 13,             # Wyjściowa liczba wymiarów MFCC (cech)
    'window': 'hamming',      # Rodzaj okna
    }
    fs = 16000

    path = "./Plots/MFCCs"
    if not os.path.exists(path):
        os.mkdir(path)

    for index, row in data[data['Hel'] == False].iterrows():
        figure, ax = plt.subplots(2, 1, figsize=(10, 18))

        row_hel = data[(data['Hel'] == True) &
                       (data['Content'] == row['Content']) &
                       (data['Speaker'] == row['Speaker'])]

        if len(row['Signal']) > len(row_hel['Signal'].iloc[0]):
            s_nohel = (row['Signal'])[:(len(row_hel['Signal'].iloc[0]))]
            s_hel = row_hel['Signal'].iloc[0]
        else:
            s_nohel = row['Signal']
            s_hel = (row_hel['Signal'].iloc[0])[:len(row['Signal'])]

        MFCC_nohel = librosa.feature.mfcc(y=s_nohel.astype(float), S=None, sr=fs, **args)
        MFCC_hel  = librosa.feature.mfcc(y=s_hel.astype(float), S=None, sr=fs, **args)

        t_nohel, f_nohel = np.arange(1, np.shape(MFCC_nohel)[0]), np.arange(0, np.shape(MFCC_nohel)[1])
        t_hel, f_hel = np.arange(1, np.shape(MFCC_hel)[0]), np.arange(0, np.shape(MFCC_hel)[1])

        xmin, xmax = f_nohel.min(), f_nohel.max()
        ymin, ymax = t_nohel.min(), t_nohel.max()
        im = ax[0].imshow(MFCC_nohel, extent=[xmin, xmax, ymin, ymax],
                         aspect='auto', origin='lower', interpolation='none')
        ax[0].set_xlabel("Time [sec]")
        ax[0].set_ylabel("# Coeff.")
        ax[0].set_title(f"{row['Speaker']}, {row['Content']}, NoHel")
        ax[0].set_yticks(t_nohel)
        figure.colorbar(im, ax=ax[0])

        xmin, xmax = f_hel.min(), f_hel.max()
        ymin, ymax = t_hel.min(), t_hel.max()
        im = ax[1].imshow(MFCC_hel, extent=[xmin, xmax, ymin, ymax],
                         aspect='auto', origin='lower', interpolation='none')
        ax[1].set_xlabel("Time [sec]")
        ax[1].set_ylabel("# Coeff.")
        ax[1].set_title(f"{row['Speaker']}, {row['Content']}, Hel")
        ax[1].set_yticks(t_hel)
        figure.colorbar(im, ax=ax[1])

        plt.savefig("./Plots/MFCCs/" + f"MFCC_{row['Speaker']}_{row['Content']}.png")
        plt.close()
    return
