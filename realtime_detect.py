import numpy as np
import sounddevice as sd
import tensorflow as tf
from audio_utils import extract_mfcc
import threading
import queue
import time
import os
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import librosa

# Constants
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.3
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
THRESHOLD = 0.85
COOLDOWN_PERIOD = 1.0
MIN_AUDIO_LEVEL = 0.05
NUM_MFCC = 13
NUM_FRAMES = 32
MIN_CONFIDENCE_DISPLAY = 0.85
SPEAKER_THRESHOLD = 0.80

# Global variables
last_detection_time = 0
model = None
audio_queue = queue.Queue()
is_speaking = False
speech_lock = threading.Lock()
is_in_cooldown = False
last_speech_time = 0
last_prediction = 0
consecutive_high_confidence = 0
speaker_embeddings = []
current_speaker = None
response_lock = threading.Lock()  # New lock for response synchronization

def speak_response():
    """Speak the response using macOS say command"""
    global is_speaking, is_in_cooldown, last_speech_time
    
    # Try to acquire the response lock
    if not response_lock.acquire(blocking=False):
        print("Another response is in progress, skipping...")
        return
        
    try:
        with speech_lock:
            if is_speaking:  # Skip if already speaking
                return
            is_speaking = True
            
        print("\n=== STARTING RESPONSE ===")
        
        # Try different voices if one fails
        voices = ['Samantha', 'Alex', 'Daniel', 'Karen']
        success = False
        
        for voice in voices:
            try:
                print(f"Trying voice: {voice}")
                # Run the say command
                subprocess.run(['say', '-v', voice, 'Hello there, how can I help you?'], 
                             check=True)
                print(f"Response complete with voice: {voice}")
                success = True
                break
            except subprocess.CalledProcessError as e:
                print(f"Failed with voice {voice}: {str(e)}")
                continue
        
        if not success:
            # Last resort: try without specifying voice
            try:
                print("Trying default voice...")
                subprocess.run(['say', 'Hello there, how can I help you?'], 
                             check=True)
                print("Response complete with default voice")
            except subprocess.CalledProcessError as e:
                print(f"All voice attempts failed: {str(e)}")
        
    except Exception as e:
        print(f"Error in speak_response: {str(e)}")
    finally:
        with speech_lock:
            is_speaking = False
        response_lock.release()
        print("=== RESPONSE COMPLETE ===\n")

def load_model():
    """Load the trained model"""
    global model
    try:
        print("[SYSTEM] Loading wake word model...")
        model = tf.keras.models.load_model('models/wake_word_model.h5')
        print("[SYSTEM] Model loaded successfully!")
        
        # Test the model with a simple input
        test_input = np.zeros((1, NUM_FRAMES, NUM_MFCC, 1))
        test_output = model.predict(test_input, verbose=0)
        print(f"[SYSTEM] Model test successful. Output shape: {test_output.shape}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        print("[ERROR] Please ensure the model file exists at 'models/wake_word_model.h5'")
        raise

def load_speaker_samples():
    """Load and process speaker samples from WAV files"""
    print("\nLoading speaker samples...")
    samples_dir = 'data/oknino'  # Updated directory path
    all_mfccs = []
    
    try:
        # Get all WAV files
        wav_files = [f for f in os.listdir(samples_dir) if f.endswith('.wav')]
        
        if not wav_files:
            print("No WAV files found in the oknino directory. Please ensure the files are extracted there.")
            return
            
        for wav_file in wav_files:
            file_path = os.path.join(samples_dir, wav_file)
            try:
                # Load audio file
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Extract MFCC features
                mfcc = extract_mfcc(audio, SAMPLE_RATE, NUM_MFCC, NUM_FRAMES)
                all_mfccs.append(mfcc)
                print(f"Processed {wav_file}")
                
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")
                continue
        
        if all_mfccs:
            # Save all speaker embeddings
            global speaker_embeddings
            speaker_embeddings = all_mfccs
            save_speaker_embeddings()
            print(f"\nSuccessfully processed {len(all_mfccs)} speaker samples")
        else:
            print("No valid speaker samples found!")
            
    except Exception as e:
        print(f"Error loading speaker samples: {str(e)}")

def save_speaker_embeddings():
    """Save speaker embeddings to file"""
    try:
        with open('speaker_embeddings.pkl', 'wb') as f:
            pickle.dump(speaker_embeddings, f)
        print("Speaker embeddings saved successfully")
    except Exception as e:
        print(f"Error saving speaker embeddings: {str(e)}")

def enroll_speaker(mfcc):
    """Enroll a new speaker"""
    global speaker_embeddings
    speaker_embeddings.append(mfcc)
    save_speaker_embeddings()
    print("New speaker enrolled successfully")

def is_known_speaker(mfcc, threshold=SPEAKER_THRESHOLD):
    """Check if the current audio matches any known speaker"""
    # Speaker recognition disabled: always return True
    return True

def process_audio(audio_chunk, model):
    """Process audio chunk and detect wake word"""
    global last_detection_time, is_speaking, is_in_cooldown, last_speech_time, last_prediction, consecutive_high_confidence, current_speaker
    
    try:
        current_time = time.time()
        
        # Skip if speaking or in cooldown
        if is_speaking or is_in_cooldown:
            if current_time - last_speech_time >= COOLDOWN_PERIOD:
                is_in_cooldown = False
                consecutive_high_confidence = 0
            return
            
        # Check audio level
        audio_level = np.max(np.abs(audio_chunk))
        if audio_level < MIN_AUDIO_LEVEL:
            consecutive_high_confidence = 0
            return
        
        # Extract MFCC features
        mfcc = extract_mfcc(audio_chunk, SAMPLE_RATE, NUM_MFCC, NUM_FRAMES)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)
        
        # Make prediction
        prediction = model.predict(mfcc, verbose=0)[0][0]
        
        # Apply prediction smoothing
        smoothed_prediction = 0.8 * prediction + 0.2 * last_prediction
        last_prediction = prediction
        
        # Track high confidence frames - more strict counting
        if smoothed_prediction > THRESHOLD:
            consecutive_high_confidence += 1
            # Only count extra frames for very high confidence
            if smoothed_prediction > 0.98:
                consecutive_high_confidence += 1
        else:
            consecutive_high_confidence = max(0, consecutive_high_confidence - 1)  # Gradual decrease
        
        # Print confidence if high enough and audio level is good
        if smoothed_prediction > MIN_CONFIDENCE_DISPLAY and audio_level > MIN_AUDIO_LEVEL * 1.2:
            print(f"Confidence: {smoothed_prediction:.2%} (Frames: {consecutive_high_confidence}) (Audio: {audio_level:.2f})", end='\r')
        else:
            print("Listening...", end='\r')
        
        # Wake word detection - require more consecutive high confidence frames
        if (smoothed_prediction > THRESHOLD and 
            not is_in_cooldown and 
            not is_speaking and  # Double check not speaking
            consecutive_high_confidence >= 1 and  # Lowered from 2 to 1 for easiest detection
            audio_level > MIN_AUDIO_LEVEL * 1.2):
            
            # Check if the speaker is recognized
            if not is_known_speaker(mfcc[0]):
                print("\nWake word detected but speaker not recognized!")
                consecutive_high_confidence = 0
                return
                
            print("\nWake word 'Ok Nino' detected from recognized speaker!")
            print("Response: Hello there, how can I help you?\n")
            print(f"Debug - Confidence: {smoothed_prediction:.2%}")
            print(f"Debug - Consecutive frames: {consecutive_high_confidence}")
            print(f"Debug - Audio level: {audio_level:.2f}")
            
            # Set cooldown immediately
            is_in_cooldown = True
            last_speech_time = current_time
            last_detection_time = current_time
            consecutive_high_confidence = 0
            
            # Call speak_response directly
            print("Calling speak_response...")
            speak_response()

    except Exception as e:
        print(f"Error processing audio: {str(e)}")

def audio_callback(indata, frames, time, status, audio_queue):
    """Callback function for audio stream"""
    if status:
        print(f"Status: {status}")
    try:
        audio_queue.put(indata.copy())
    except Exception as e:
        print(f"Error in audio callback: {str(e)}")

def main():
    print("Starting wake word detection...")
    print("Say 'Ok Nino' to activate the assistant")
    print("Press Ctrl+C to stop")
    print("\nListening... (only responds to 'Ok Nino' from recognized speakers)")
    
    # Test TTS at startup
    try:
        print("\n=== TESTING TTS SYSTEM ===")
        print("Testing TTS system...")
        # Try different voices
        voices = ['Samantha', 'Alex', 'Daniel', 'Karen']
        success = False
        
        for voice in voices:
            try:
                print(f"Testing voice: {voice}")
                subprocess.run(['say', '-v', voice, 'System initialized'], check=True)
                print(f"TTS test successful with voice: {voice}")
                success = True
                break
            except subprocess.CalledProcessError as e:
                print(f"Failed with voice {voice}: {str(e)}")
                continue
        
        if not success:
            # Last resort: try without specifying voice
            try:
                print("Testing default voice...")
                subprocess.run(['say', 'System initialized'], check=True)
                print("TTS test successful with default voice")
            except subprocess.CalledProcessError as e:
                print(f"All voice attempts failed: {str(e)}")
                return
        
        print("=== TTS TEST COMPLETE ===\n")
    except Exception as e:
        print(f"Warning: TTS test failed: {str(e)}")
        return
    
    # Load the model and speaker embeddings
    try:
        model = load_model()
        if model is None:
            print("Failed to load model. Exiting...")
            return
            
        load_speaker_samples()
        if not speaker_embeddings:
            print("Warning: No speaker samples loaded. Will respond to any voice.")
        else:
            print(f"Loaded {len(speaker_embeddings)} speaker samples.")
    except Exception as e:
        print(f"Failed to initialize: {str(e)}")
        return
    
    try:
        # Start audio stream
        with sd.InputStream(callback=lambda indata, frames, time, status: 
                           audio_callback(indata, frames, time, status, audio_queue),
                           channels=1,
                           samplerate=SAMPLE_RATE,
                           blocksize=CHUNK_SIZE):
            
            while True:
                try:
                    # Get audio chunk from queue
                    audio_chunk = audio_queue.get()
                    
                    # Process audio
                    process_audio(audio_chunk.flatten(), model)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[ERROR] Error in main loop: {str(e)}")
    
    finally:
        print("\nStopping wake word detection...")

if __name__ == "__main__":
    main() 