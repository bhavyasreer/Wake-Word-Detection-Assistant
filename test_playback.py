import numpy as np
import tensorflow as tf
from audio_utils import extract_mfcc, load_audio
import os
import sounddevice as sd
import time
import pyttsx3

def init_tts():
    """Initialize the TTS engine"""
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)    # Speed of speech
    tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # Try to set a female voice if available
    voices = tts_engine.getProperty('voices')
    for voice in voices:
        if 'female' in voice.name.lower():
            tts_engine.setProperty('voice', voice.id)
            break
    return tts_engine

def speak_response(text):
    """Speak the response"""
    try:
        tts_engine = init_tts()
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error in speech: {str(e)}")

def test_playback(model_path='models/wake_word_model.h5', 
                 wake_word_dir='data/wake_word',
                 background_dir='data/background',
                 threshold=0.95):
    """
    Test the model by playing back recorded samples
    """
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    print("\nLoading test data...")
    wake_word_files = [f for f in os.listdir(wake_word_dir) if f.endswith('.wav')]
    background_files = [f for f in os.listdir(background_dir) if f.endswith('.wav')]
    
    print("\nTesting wake word samples...")
    for i, file in enumerate(wake_word_files, 1):
        audio_path = os.path.join(wake_word_dir, file)
        try:
            # Load audio
            audio = load_audio(audio_path)
            
            # Play the audio
            print(f"\nPlaying wake word sample {i}/{len(wake_word_files)}...")
            sd.play(audio, 16000)
            sd.wait()
            
            # Process audio
            mfcc = extract_mfcc(audio)
            mfcc = np.expand_dims(mfcc, axis=0)
            mfcc = np.expand_dims(mfcc, axis=-1)
            
            # Make prediction
            prediction = model.predict(mfcc, verbose=0)[0][0]
            
            # Print result and respond
            if prediction > threshold:
                print(f"✓ Wake word detected! (Confidence: {prediction:.2%})")
                print("[RESPONSE] Hello there, how can I help you?")
                speak_response("Hello there, how can I help you?")
            else:
                print(f"✗ Wake word NOT detected (Confidence: {prediction:.2%})")
            
            # Wait a bit between samples
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    print("\nTesting background samples...")
    for i, file in enumerate(background_files, 1):
        audio_path = os.path.join(background_dir, file)
        try:
            # Load audio
            audio = load_audio(audio_path)
            
            # Play the audio
            print(f"\nPlaying background sample {i}/{len(background_files)}...")
            sd.play(audio, 16000)
            sd.wait()
            
            # Process audio
            mfcc = extract_mfcc(audio)
            mfcc = np.expand_dims(mfcc, axis=0)
            mfcc = np.expand_dims(mfcc, axis=-1)
            
            # Make prediction
            prediction = model.predict(mfcc, verbose=0)[0][0]
            
            # Print result
            if prediction > threshold:
                print(f"✗ False alarm! (Confidence: {prediction:.2%})")
                print("[RESPONSE] Hello there, how can I help you?")
                speak_response("Hello there, how can I help you?")
            else:
                print(f"✓ Correctly ignored background (Confidence: {prediction:.2%})")
            
            # Wait a bit between samples
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    test_playback() 