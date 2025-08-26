import numpy as np
import sounddevice as sd
import tensorflow as tf
from audio_utils import extract_mfcc
import threading
import queue
import time
import os
import pyttsx3
import platform

# Constants
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
THRESHOLD = 0.95
COOLDOWN_PERIOD = 0.3
MIN_CONFIDENCE = 0.6
MIN_AUDIO_LEVEL = 0.5
LISTENING_WINDOW = 5.0

# Global variables
last_detection_time = 0
is_listening_for_command = False
listening_start_time = 0
model = None
audio_queue = queue.Queue()
last_prediction = 0
silence_counter = 0
tts_engine = None
is_running = False
callback_function = None

def set_callback(callback):
    """Set the callback function to be called when wake word is detected"""
    global callback_function
    callback_function = callback

def start_detection():
    """Start the wake word detection"""
    global is_running, model
    
    if is_running:
        return
        
    is_running = True
    print("Starting wake word detection...")
    
    # Load the model
    if model is None:
        model = tf.keras.models.load_model('models/wake_word_model.h5')
    
    # Start audio stream
    with sd.InputStream(callback=lambda indata, frames, time, status: 
                       audio_callback(indata, frames, time, status, audio_queue),
                       channels=1,
                       samplerate=SAMPLE_RATE,
                       blocksize=CHUNK_SIZE):
        
        while is_running:
            try:
                # Get audio chunk from queue
                audio_chunk = audio_queue.get()
                
                # Process audio
                process_audio(audio_chunk.flatten(), model)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

def stop_detection():
    """Stop the wake word detection"""
    global is_running
    is_running = False
    print("Stopping wake word detection...")

def process_audio(audio_chunk, model):
    """Process audio chunk and detect wake word"""
    global last_detection_time, is_listening_for_command, listening_start_time, last_prediction, silence_counter, callback_function
    
    try:
        # First check if this is silence
        if is_silence(audio_chunk):
            silence_counter += 1
            if silence_counter > 10:
                return
        else:
            silence_counter = 0
        
        if silence_counter > 0:
            return
            
        # Preprocess audio
        audio_chunk = preprocess_audio(audio_chunk)
        
        # Extract MFCC features
        mfcc = extract_mfcc(audio_chunk, SAMPLE_RATE)
        
        # Add batch and channel dimensions
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)
        
        # Make prediction
        prediction = model.predict(mfcc, verbose=0)[0][0]
        
        # Apply prediction smoothing
        smoothed_prediction = 0.8 * prediction + 0.2 * last_prediction
        last_prediction = prediction
        
        current_time = time.time()
        
        if is_listening_for_command:
            if current_time - listening_start_time > LISTENING_WINDOW:
                is_listening_for_command = False
                print("\nListening window expired. Waiting for wake word...")
            return
        
        # Check for wake word
        if (smoothed_prediction > THRESHOLD and 
            (current_time - last_detection_time) >= COOLDOWN_PERIOD and
            np.max(np.abs(audio_chunk)) > MIN_AUDIO_LEVEL * 3.0 and
            silence_counter == 0):
            
            print(f"\nWake word detected! Confidence: {smoothed_prediction:.2%}")
            last_detection_time = current_time
            
            # Call the callback function if set
            if callback_function:
                callback_function()
            
            # Start listening window
            is_listening_for_command = True
            listening_start_time = current_time
            print("\nListening for command (5 seconds)...")

    except Exception as e:
        print(f"Error processing audio: {str(e)}")

def audio_callback(indata, frames, time, status, audio_queue):
    """Callback function for audio stream"""
    if status:
        print(f"Status: {status}")
    audio_queue.put(indata.copy())

def is_silence(audio_chunk):
    """Check if the audio chunk is silence"""
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_chunk))))
    zero_crossing_rate = zero_crossings / len(audio_chunk)
    peak_amplitude = np.max(np.abs(audio_chunk))
    
    return (rms < 0.02 or 
            zero_crossing_rate < 0.05 or 
            peak_amplitude < MIN_AUDIO_LEVEL)

def preprocess_audio(audio):
    """Preprocess audio to reduce noise"""
    audio = audio / np.max(np.abs(audio))
    noise_floor = 0.05
    audio[np.abs(audio) < noise_floor] = 0
    return audio 