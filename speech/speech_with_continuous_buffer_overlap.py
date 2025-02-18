import speech_recognition as sr
import whisper
import numpy as np
import threading
import datetime
import time
import noisereduce as nr
import webrtcvad
import pyaudio
from collections import deque
import difflib

#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.max_memory_allocated())
#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name())

recognizer = sr.Recognizer()
model = whisper.load_model("large")
microphone_index = 2

# Audio recording parameters
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate in Hz

chunk_duration = 5  # seconds
overlap_duration = 1  # seconds
chunk_frames = int(RATE * chunk_duration)
overlap_frames = int(RATE * overlap_duration)

vad = webrtcvad.Vad()
vad.set_mode(1)
sample_rate = 16000

stop_event = threading.Event()
previous_transcription = ''

# Initialize PyAudio
p = pyaudio.PyAudio()

# Initialize audio buffer
audio_buffer = deque(maxlen=chunk_frames + overlap_frames)

def audio_callback(in_data, frame_count, time_info, status):
    if stop_event.is_set():
        return (None, pyaudio.paComplete)
    # Convert byte data to numpy array
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    audio_buffer.extend(audio_data)
    return (in_data, pyaudio.paContinue)

def start_stream():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)
    p.get_default_input_device_info()
    stream.start_stream()
    print("Listening...")
    return stream

def merge_transcriptions_using_difflib(prev, curr):
    sequence_matcher = difflib.SequenceMatcher(None, prev, curr)
    match = sequence_matcher.find_longest_match(0, len(prev), 0, len(curr))
    if match.size > 0:
        return prev + curr[match.size:]
    else:
        return prev + " " + curr

def listen_and_transcribe():
        try:
            while not stop_event.is_set():
                if len(audio_buffer) >= chunk_frames + overlap_frames:
                    # Extract frames for transcription
                    print("Extracting...")
                    frames_to_transcribe = list(audio_buffer)
                    # Remove the frames we just took, keeping the overlap
                    for _ in range(chunk_frames):
                        audio_buffer.popleft()
                    # Transcribe in a new thread
                    threading.Thread(target=transcribe_audio, args=(frames_to_transcribe,), daemon=True).start()
                else:
                    # Wait for more data
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping listener thread...")
            stream.stop_stream()
            stream.close()
            p.terminate()

def transcribe_audio(audio):
    start_time = datetime.datetime.now()
    print("Transcribing...")
    global previous_transcription
    #print(f"[{start_time.strftime('%H:%M:%S')}] Beginning Transcribe.", flush=True)
    try:

        # Fetch raw audio data in 16-bit format
        #audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_data = audio
        
        # Convert byte data to a NumPy array of int16
        #audio_np_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_np = np.array(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Normalize and convert to float32 in the range [-1.0, 1.0]
        #audio_float32 = audio_np_int16.astype(np.float32) / 32768.0

        reduced_noise = nr.reduce_noise(y=audio_np, sr=16000)
        
        result = model.transcribe(reduced_noise, fp16=False, language='en', task='transcribe', condition_on_previous_text=True)

        #print(f"[{start_time.strftime('%H:%M:%S')}] Ending Transcribe.", flush=True)

        transcription = result['text']

        if transcription:
            combined_transcription = merge_transcriptions_using_difflib(previous_transcription, transcription.strip())
            print(f"[{start_time.strftime('%H:%M:%S')}] Current Transcription:",  transcription, flush=True)
            print(f"[{start_time.strftime('%H:%M:%S')}] Previous Transcription:",  previous_transcription, flush=True)
            print(f"[{start_time.strftime('%H:%M:%S')}] Combined Transcription:",  combined_transcription, flush=True)
            previous_transcription = combined_transcription
        else:
            print(f"[{start_time.strftime('%H:%M:%S')}] No speech detected.", flush=True)
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")

    end_time = datetime.datetime.now()
    
    print(f"Transcription time: {(end_time - start_time).total_seconds()} seconds", flush=True)

if __name__ == "__main__":
    # Start audio stream
    stream = start_stream()
    # Start processing audio
    listening_thread = threading.Thread(target=listen_and_transcribe)
    listening_thread.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping main thread...")
        stop_event.set()
        listening_thread.join()
        stream.stop_stream()
        stream.close()
        p.terminate()