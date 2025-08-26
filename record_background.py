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
    
    print("Recording background noise samples")
    print("Press Enter to start recording each sample")
    print("Press Ctrl+C to stop")
    print("\nRecord different types of background noise:")
    print("1. Room ambience")
    print("2. Keyboard typing")
    print("3. Mouse clicks")
    print("4. Paper rustling")
    print("5. Chair movement")
    print("6. Door closing")
    print("7. Footsteps")
    print("8. General household sounds")
    
    try:
        sample_count = 0
        while True:
            input(f"\nPress Enter to record background sample {sample_count + 1}")
            audio = record_audio()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"background_{timestamp}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Save the recording
            save_audio(audio, filepath)
            print(f"Saved as: {filename}")
            sample_count += 1
            
    except KeyboardInterrupt:
        print(f"\nRecording stopped. Recorded {sample_count} background samples.")

if __name__ == "__main__":
    main() 