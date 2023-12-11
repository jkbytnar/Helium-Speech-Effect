import librosa
import numpy as np
import soundfile as sf


def load_file(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr


def my_stft(signal, sr):
    stft_matrix = librosa.stft(signal)
    return stft_matrix


def modify_stft(stft_matrix):
    modified_stft = 3 * stft_matrix  # Example operation - multiply by 3
    return modified_stft


def my_istft(modified_stft):
    modified_signal = librosa.istft(modified_stft)
    return modified_signal


def save_to_file(modified_signal, sr, file_name):
    sf.write(file_name, modified_signal, sr)


def main(input_file_path, output_file_name):
    signal, sr = load_file(input_file_path)
    stft_matrix = my_stft(signal, sr)
    modified_stft = modify_stft(stft_matrix)
    modified_signal = my_istft(modified_stft)
    save_to_file(modified_signal, sr, output_file_name)


if __name__ == "__main__":
    input_file_path = 'A_Hel.wav'
    output_file_name = 'modified_A_Hel.wav'
    main(input_file_path, output_file_name)
