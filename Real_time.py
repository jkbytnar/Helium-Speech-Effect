import sounddevice as sd
import numpy as np
import Hellium
import wave
from scipy.io.wavfile import write

hel_glob = np.array([])
in_glob = np.array([])
hel_scaled_glob = np.array([])
def print_sound(indata, outdata,  frames, time, status):
    global hel_glob, in_glob, hel_scaled_glob
    in_data_gain = np.mean(indata)
    hel = Hellium.voice2hel(indata)
    hel_gain = np.mean(hel)
    scale = in_data_gain/hel_gain
    hel_scaled = hel*scale

    hel_glob = np.concatenate((hel_glob, hel[0]))
    in_glob = np.concatenate((in_glob, indata[0]))
    hel_scaled_glob = np.concatenate((hel_scaled_glob, hel_scaled[0]))
    outdata[:] = indata

try:

    with sd.Stream(channels=1, samplerate=16000, callback=print_sound):
        input()
        print(np.shape(in_glob))

        channels = 1
        sample_width = 4  # 4 bajty dla 32-bit PCM
        output_file = 'output_in.wav'
        # Tworzymy obiekt wave
        with wave.open(output_file, 'w') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(16000)
            wav_file.writeframes(in_glob.astype(np.float32).tobytes())

        output_file = 'output_hel.wav'
        # Tworzymy obiekt wave
        with wave.open(output_file, 'w') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(16000)
            wav_file.writeframes(hel_scaled_glob.astype(np.float32).tobytes())

        output_file = 'output_hel_scaled.wav'
        # Tworzymy obiekt wave
        with wave.open(output_file, 'w') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(16000)
            wav_file.writeframes(hel_glob.astype(np.float32).tobytes())
except KeyboardInterrupt:
    exit()