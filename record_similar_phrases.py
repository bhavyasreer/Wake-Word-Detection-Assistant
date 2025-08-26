import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime

# Constants
SAMPLE_RATE = 16000
DURATION = 1.0  # Recording duration in seconds
OUTPUT_DIR = 'data/background'

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """Record audio from microphone"""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def save_audio(audio, file_path, sample_rate=SAMPLE_RATE):
    """Save audio to file"""
    sf.write(file_path, audio, sample_rate)

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Phrases to record
    phrases = [
        "Hi Nino",
        "Hello Nino",
        "Hey Nino",
        "Good morning Nino",
        "Good night Nino",
        "Bye Nino",
        "Thank you Nino",
        "Please Nino",
        "Nino please",
        "Nino help"
    ]
    
    print("Recording similar phrases as negative examples")
    print("Press Enter to start recording each phrase")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            for phrase in phrases:
                input(f"\nPress Enter to record: '{phrase}'")
                audio = record_audio()
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"similar_phrase_{timestamp}.wav"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # Save the recording
                save_audio(audio, filepath)
                print(f"Saved as: {filename}")
                
    except KeyboardInterrupt:
        print("\nRecording stopped")

if __name__ == "__main__":
    main() 