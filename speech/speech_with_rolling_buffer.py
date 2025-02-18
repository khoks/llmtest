import speech_recognition as sr
import whisper
import numpy as np
import threading
import datetime
import time
import noisereduce as nr
import webrtcvad

#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.max_memory_allocated())
#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name())

recognizer = sr.Recognizer()
model = whisper.load_model("base")
microphone_index = 2
chunk_duration = 5  # seconds
max_buffer_duration  = 15  # seconds
sample_rate = 16000
vad = webrtcvad.Vad()
vad.set_mode(1)

def listen_and_transcribe():
    with sr.Microphone(device_index=microphone_index) as source:

        audio_buffer = b''

        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with index {index}: {name}")
        print(source.list_working_microphones())

        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening continuously. Press Ctrl+C to stop.")
        try:
            while True:
                start_time = time.time()

                audio = recognizer.listen(source, phrase_time_limit=chunk_duration)

                audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                # Combine overlap with current audio_data
                combined_audio = audio_buffer + audio_data

                threading.Thread(target=transcribe_audio, args=(combined_audio,)).start()
                #transcribe_audio(audio)

                audio_buffer += audio_data
                max_buffer_size = int(max_buffer_duration * sample_rate * source.SAMPLE_WIDTH)
                # Keep only the last 'overlap_duration' seconds in the buffer
                if len(audio_buffer) > max_buffer_size:
                    audio_buffer = audio_buffer[-max_buffer_size:]

        except KeyboardInterrupt:
            print("\nStopped listening.")

def transcribe_audio(audio):
    start_time = datetime.datetime.now()
    #print(f"[{start_time.strftime('%H:%M:%S')}] Beginning Transcribe.", flush=True)

    # Fetch raw audio data in 16-bit format
    #audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
    audio_data = audio
    
    # Convert byte data to a NumPy array of int16
    audio_np_int16 = np.frombuffer(audio_data, dtype=np.int16)

    # Normalize and convert to float32 in the range [-1.0, 1.0]
    audio_float32 = audio_np_int16.astype(np.float32) / 32768.0

    reduced_noise = nr.reduce_noise(y=audio_float32, sr=16000)
    
    result = model.transcribe(reduced_noise, fp16=False, language='en', task='transcribe', condition_on_previous_text=True)

    #print(f"[{start_time.strftime('%H:%M:%S')}] Ending Transcribe.", flush=True)

    transcription = result['text']

    if transcription:
        print(f"[{start_time.strftime('%H:%M:%S')}] Transcription:",  transcription.strip(), flush=True)
    else:
        print(f"[{start_time.strftime('%H:%M:%S')}] No speech detected.", flush=True)

    end_time = datetime.datetime.now()
    
    print(f"Transcription time: {(end_time - start_time).total_seconds()} seconds", flush=True)

listen_and_transcribe()