import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np

# Recording parameters
fs = 16000  # Sample rate
seconds = 5  # Duration of recording

print("Recording...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
print("Recording complete.")

# Save the recording to a WAV file
write('output.wav', fs, myrecording)

# Playback the recording
print("Playing back the recording...")
sd.play(myrecording, fs)
sd.wait()  # Wait until playback is finished
print("Playback complete.")