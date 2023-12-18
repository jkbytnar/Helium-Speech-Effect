import pyaudio
import numpy as np
import helium
import keyboard

chunk = 1024*8
audio_format = pyaudio.paFloat32
channels = 1
rate = 16000
p = pyaudio.PyAudio()

stream = p.open(
    format=audio_format,
    channels=channels,
    rate=rate,
    input=True,
    frames_per_buffer=chunk
)

player = p.open(
    format=audio_format,
    channels=channels,
    rate=rate,
    output=True,
    frames_per_buffer=chunk
)

while True:
    try:
        data = np.frombuffer(stream.read(chunk), dtype=np.float32)
        audio_hel = helium.voice2hel(data)
        player.write(audio_hel, chunk)
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))

    if keyboard.is_pressed('Esc'):
        print('\nInterrupted by user')
        break

stream.stop_stream()
stream.close()
p.terminate()
