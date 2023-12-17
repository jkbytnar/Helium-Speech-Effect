import pyaudio
import numpy as np
import Helium

chunk = 1024*4
format = pyaudio.paFloat32
channels = 1
rate = 16000
p = pyaudio.PyAudio()

stream = p.open(
    format=format,
    channels=channels,
    rate=rate,
    input=True,
    frames_per_buffer=chunk
)

player = p.open(
    format=format,
    channels=channels,
    rate=rate,
    output=True,
    frames_per_buffer=chunk
)

while True:
    data = np.fromstring(stream.read(chunk),dtype=np.float32)
    helium = Helium.voice2hel(data)
    player.write(helium, chunk)

stream.stop_stream()
stream.close()
p.terminate()