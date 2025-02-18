import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt

# Use the correct microphone index
microphone_index = 2

recognizer = sr.Recognizer()

def visualize_audio(audio_np):
    plt.figure(figsize=(10, 4))
    plt.plot(audio_np)
    plt.title('Audio Waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

with sr.Microphone(device_index=microphone_index) as source:
    print(f"Using microphone: {sr.Microphone.list_microphone_names()[microphone_index]}")
    print("Adjusting for ambient noise... Please wait.")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Please say something...")

    # Capture audio
    audio = recognizer.listen(source)
    print("Audio captured.")

    audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)

    # Convert byte data to a NumPy array of int16
    audio_np_int16 = np.frombuffer(audio_data, dtype=np.int16)

    # Visualize the waveform
    visualize_audio(audio_np_int16)

try:
    # Use Google's speech recognition as a quick test
    text = recognizer.recognize_google(audio)
    print("You said:", text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio.")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")