import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display


def load_file(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

def my_stft(signal, sr):
    stft_matrix = librosa.stft(signal)
    return stft_matrix

def modify_stft(stft_matrix, factor=0.49):
    original_rows, original_cols = stft_matrix.shape
    new_rows = int(original_rows * factor)
    expanded_stft = np.zeros((new_rows, original_cols), dtype=np.complex)

    for i in range(original_cols):
        expanded_stft[:, i] = np.interp(np.arange(0, new_rows, 1), np.arange(0, original_rows), stft_matrix[:, i])

    return expanded_stft

def my_istft(modified_stft):
    modified_signal = librosa.istft(modified_stft)
    return modified_signal

def save_to_file(modified_signal, sr, file_name):
    sf.write(file_name, modified_signal, sr)

def plot_spectrogram(stft_matrix, title, sr):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def main(input_file_path, output_file_name):
    input_signal, sr = load_file(input_file_path)
    
    input_stft = my_stft(input_signal, sr)
    plot_spectrogram(input_stft, 'Input Spectrogram', sr)

    modified_stft = modify_stft(input_stft)

    modified_signal = my_istft(modified_stft)
    plot_spectrogram(modified_stft, 'Output Spectrogram', sr)

    save_to_file(modified_signal, sr, output_file_name)


if __name__ == "__main__":
    input_file_path = 'A_NoHell.wav'
    output_file_name = 'modified_A_noHell.wav'
    main(input_file_path, output_file_name)
